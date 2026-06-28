import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor

from model import AIFaceDetector


MIN_RESOLUTION = 64
BLUR_THRESHOLD = 100.0
CONFIDENCE_THRESHOLD = 0.5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AIFaceDetector(freeze_early=True).to(device)
model.load_state_dict(torch.load(
    'models/best_model.pth', map_location=device))
model.eval()



class AgentState(TypedDict):
    
    image_path: str
    image_pil: Any

    
    quality_passed: bool
    quality_warning: str

    
    is_ai: bool
    confidence: float

    
    texture_score: float
    edge_score: float
    color_score: float

    
    verdict: str
    explanation: str



def check_blur(image_gray):
    """Laplacian variance — low value = blurry image."""
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def check_resolution(image_pil):
    """Check if image meets minimum resolution."""
    w, h = image_pil.size
    return min(w, h) >= MIN_RESOLUTION


def calculate_texture_score(image_gray):
    """
    Measure texture inconsistency.
    Splits image into patches, calculates variance of each patch.
    Unnaturally uniform variance = potentially AI generated.
    Returns score 0-100 (higher = more suspicious)
    """
    h, w = image_gray.shape
    patch_size = h // 4
    variances = []

    for i in range(4):
        for j in range(4):
            patch = image_gray[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]
            variances.append(np.var(patch))

    
    variance_of_variances = np.var(variances)
    
    score = max(0, min(100, 100 - (variance_of_variances / 500)))
    return round(score, 1)


def calculate_edge_score(image_gray):
    """
    Measure edge sharpness anomaly.
    Compares center vs border sharpness.
    AI images often have inconsistent sharpness across regions.
    Returns score 0-100 (higher = more suspicious)
    """
    h, w = image_gray.shape
    center = image_gray[h//4:3*h//4, w//4:3*w//4]
    border = np.concatenate([
        image_gray[:h//4].flatten(),
        image_gray[3*h//4:].flatten()
    ])

    center_sharpness = cv2.Laplacian(center, cv2.CV_64F).var()
    border_sharpness = np.var(cv2.Laplacian(
        border.reshape(-1, 1).astype(np.uint8), cv2.CV_64F))

    
    diff = abs(center_sharpness - border_sharpness)
    score = max(0, min(100, diff / 100))
    return round(score, 1)


def calculate_color_score(image_rgb):
    """
    Measure color distribution anomaly.
    AI images often have unnatural color clustering.
    Returns score 0-100 (higher = more suspicious)
    """
    r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)

    
    channel_imbalance = np.std([r_std, g_std, b_std])
    score = max(0, min(100, channel_imbalance / 2))
    return round(score, 1)



def input_agent(state: AgentState) -> AgentState:
    """
    Checks image quality before detection.
    Sets quality_passed and quality_warning in state.
    """
    image_pil = Image.open(state['image_path']).convert('RGB')
    image_np = np.array(image_pil)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    warnings = []

    
    if not check_resolution(image_pil):
        w, h = image_pil.size
        warnings.append(f"Low resolution ({w}x{h})")

    
    blur_score = check_blur(image_gray)
    if blur_score < BLUR_THRESHOLD:
        warnings.append(f"Image appears blurry (score: {blur_score:.1f})")

    quality_passed = len(warnings) == 0
    quality_warning = " | ".join(warnings) if warnings else ""

    return {
        **state,
        'image_pil': image_pil,
        'quality_passed': quality_passed,
        'quality_warning': quality_warning
    }


def detection_agent(state: AgentState) -> AgentState:
    """
    Runs MobileNetV3 model on image.
    Returns is_ai and confidence score.
    """
    image_pil = state['image_pil']
    tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob = torch.sigmoid(logit).item()

    is_ai = prob > CONFIDENCE_THRESHOLD
    confidence = prob if is_ai else 1 - prob

    return {
        **state,
        'is_ai': is_ai,
        'confidence': round(confidence * 100, 1)
    }


def explanation_agent(state: AgentState) -> AgentState:
    """
    Calculates heuristic scores for explainability.
    Returns texture, edge, color scores.
    """
    image_np = np.array(state['image_pil'])
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    texture = calculate_texture_score(image_gray)
    edge = calculate_edge_score(image_gray)
    color = calculate_color_score(image_np)

    return {
        **state,
        'texture_score': texture,
        'edge_score': edge,
        'color_score': color
    }


def orchestrator(state: AgentState) -> AgentState:
    """
    Runs detection and explanation agents in parallel
    using ThreadPoolExecutor.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        detection_future = executor.submit(detection_agent, state)
        explanation_future = executor.submit(explanation_agent, state)

        detection_result = detection_future.result()
        explanation_result = explanation_future.result()

    return {
        **state,
        'is_ai': detection_result['is_ai'],
        'confidence': detection_result['confidence'],
        'texture_score': explanation_result['texture_score'],
        'edge_score': explanation_result['edge_score'],
        'color_score': explanation_result['color_score']
    }


def final_agent(state: AgentState) -> AgentState:
    """
    Combines all results into final verdict and explanation.
    """
    verdict_label = "AI Generated" if state['is_ai'] else "Real"
    confidence = state['confidence']

    
    verdict = f"VERDICT: {verdict_label} ({confidence}% confident)"

    
    if state['quality_warning']:
        verdict += f"\n⚠️ Warning: {state['quality_warning']} — results may be unreliable"

    
    total = state['texture_score'] + state['edge_score'] + state['color_score']
    if total > 0:
        t_pct = round(state['texture_score'] / total * 100)
        e_pct = round(state['edge_score'] / total * 100)
        c_pct = round(state['color_score'] / total * 100)
    else:
        t_pct, e_pct, c_pct = 33, 33, 34

    explanation = f"""
EXPLANATION:
- Texture inconsistency: {t_pct}%
- Edge sharpness anomaly: {e_pct}%
- Color distribution anomaly: {c_pct}%
"""

    return {
        **state,
        'verdict': verdict,
        'explanation': explanation
    }


def quality_router(state: AgentState) -> str:
    """Conditional edge — route based on quality check."""
    if state['quality_passed']:
        return "orchestrator"
    else:
        return "rejected"


def rejected_agent(state: AgentState) -> AgentState:
    """Handles rejected images."""
    return {
        **state,
        'verdict': f"❌ Image rejected: {state['quality_warning']}",
        'explanation': "Please upload a clearer, higher resolution image."
    }



def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("input_agent", input_agent)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("final_agent", final_agent)
    graph.add_node("rejected", rejected_agent)

    graph.set_entry_point("input_agent")

    graph.add_conditional_edges(
        "input_agent",
        quality_router,
        {
            "orchestrator": "orchestrator",
            "rejected": "rejected"
        }
    )

    graph.add_edge("orchestrator", "final_agent")
    graph.add_edge("final_agent", END)
    graph.add_edge("rejected", END)

    return graph.compile()


if __name__ == "__main__":
    pipeline = build_graph()

    import os
    test_image = os.listdir('data/real-vs-fake/test/real')[0]
    test_image = f'data/real-vs-fake/test/real/{test_image}'
    print(f"Testing with: {test_image}")

    initial_state = AgentState(
        image_path=test_image,
        image_pil=None,
        quality_passed=False,
        quality_warning="",
        is_ai=False,
        confidence=0.0,
        texture_score=0.0,
        edge_score=0.0,
        color_score=0.0,
        verdict="",
        explanation=""
    )

    result = pipeline.invoke(initial_state)

    print(result['verdict'])
    print(result['explanation'])