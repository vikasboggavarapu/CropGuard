import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "cropguard_best.pth"
METADATA_PATH = MODEL_DIR / "metadata.json"

# ── Disease information database ──────────────────────────────────────────────
DISEASE_INFO = {
    "Corn___Common_Rust": {
        "display_name": "Corn Common Rust",
        "crop": "Corn",
        "status": "Diseased",
        "severity": "Moderate",
        "description": "Common rust is caused by the fungus Puccinia sorghi. Characterized by small, circular to elongated, cinnamon-brown pustules on both leaf surfaces.",
        "symptoms": ["Cinnamon-brown pustules on leaves", "Yellow halos around pustules", "Premature leaf senescence"],
        "treatment": "Apply foliar fungicides (triazoles, strobilurins). Ensure good air circulation. Plant resistant varieties.",
        "prevention": "Use certified disease-free seed, rotate crops, apply preventive fungicide at early stages.",
    },
    "Corn___Gray_Leaf_Spot": {
        "display_name": "Corn Gray Leaf Spot",
        "crop": "Corn",
        "status": "Diseased",
        "severity": "Moderate to High",
        "description": "Caused by Cercospora zeae-maydis fungus. A major foliar disease causing significant yield losses in high-humidity conditions.",
        "symptoms": ["Rectangular, gray to tan lesions", "Lesions bounded by leaf veins", "Lesions coalesce causing blighting"],
        "treatment": "Apply fungicides containing azoxystrobin or propiconazole. Improve field drainage.",
        "prevention": "Crop rotation, use resistant hybrids, bury crop debris after harvest.",
    },
    "Corn___Healthy": {
        "display_name": "Healthy Corn",
        "crop": "Corn",
        "status": "Healthy",
        "severity": "None",
        "description": "The corn plant shows no signs of disease. Leaves are green and vibrant with no visible lesions or discoloration.",
        "symptoms": [],
        "treatment": "No treatment needed. Continue regular agronomic practices.",
        "prevention": "Maintain balanced fertilization, proper irrigation, and regular scouting.",
    },
    "Corn___Northern_Leaf_Blight": {
        "display_name": "Corn Northern Leaf Blight",
        "crop": "Corn",
        "status": "Diseased",
        "severity": "High",
        "description": "Caused by Exserohilum turcicum. One of the most destructive diseases of corn, producing large tan lesions.",
        "symptoms": ["Large, cigar-shaped gray-green or tan lesions", "Lesions 2.5–15 cm long", "Dark sporulation in lesion center"],
        "treatment": "Foliar fungicides at tassel emergence. Azoxystrobin or propiconazole are effective.",
        "prevention": "Plant resistant hybrids, rotate with non-host crops, bury residue.",
    },
    "Potato___Early_Blight": {
        "display_name": "Potato Early Blight",
        "crop": "Potato",
        "status": "Diseased",
        "severity": "Moderate",
        "description": "Caused by Alternaria solani fungus. Appears during warm, wet weather and primarily affects older leaves first.",
        "symptoms": ["Dark brown concentric ring lesions (target board pattern)", "Yellow halo around lesions", "Lesions on older leaves first"],
        "treatment": "Apply chlorothalonil or mancozeb fungicides. Remove infected foliage.",
        "prevention": "Avoid overhead irrigation, ensure crop rotation, use certified seed tubers.",
    },
    "Potato___Healthy": {
        "display_name": "Healthy Potato",
        "crop": "Potato",
        "status": "Healthy",
        "severity": "None",
        "description": "The potato plant appears healthy with no disease symptoms present.",
        "symptoms": [],
        "treatment": "No treatment needed.",
        "prevention": "Monitor regularly, maintain adequate spacing, balanced fertilization.",
    },
    "Potato___Late_Blight": {
        "display_name": "Potato Late Blight",
        "crop": "Potato",
        "status": "Diseased",
        "severity": "Very High",
        "description": "Caused by Phytophthora infestans — the pathogen responsible for the Irish Potato Famine. Highly destructive and spreads rapidly.",
        "symptoms": ["Water-soaked lesions turning brown/black", "White mold on leaf undersides", "Rapid collapse of foliage"],
        "treatment": "Apply metalaxyl or cymoxanil-based fungicides immediately. Remove and destroy infected plants.",
        "prevention": "Use certified blight-resistant varieties, avoid overhead irrigation, scout regularly.",
    },
    "Rice___Brown_Spot": {
        "display_name": "Rice Brown Spot",
        "crop": "Rice",
        "status": "Diseased",
        "severity": "Moderate",
        "description": "Caused by Bipolaris oryzae. Associated with nutrient-deficient soils, particularly potassium deficiency.",
        "symptoms": ["Circular to oval brown spots on leaves", "Spots with gray/white center and brown margin", "Infected grains turn brown"],
        "treatment": "Apply iprodione or propiconazole fungicides. Supplement with potassium fertilizer.",
        "prevention": "Use balanced fertilization, treat seeds with thiram, avoid drought stress.",
    },
    "Rice___Healthy": {
        "display_name": "Healthy Rice",
        "crop": "Rice",
        "status": "Healthy",
        "severity": "None",
        "description": "The rice plant shows no signs of disease and is in healthy condition.",
        "symptoms": [],
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper water levels, balanced nutrition, regular field monitoring.",
    },
    "Rice___Leaf_Blast": {
        "display_name": "Rice Leaf Blast",
        "crop": "Rice",
        "status": "Diseased",
        "severity": "High",
        "description": "Caused by Magnaporthe oryzae. One of the most devastating rice diseases globally, can cause up to 100% crop loss.",
        "symptoms": ["Diamond-shaped lesions with gray-white center", "Brown to red-brown borders on lesions", "Lesions can girdle entire leaf"],
        "treatment": "Apply tricyclazole or isoprothiolane fungicides. Reduce nitrogen applications.",
        "prevention": "Plant resistant varieties, avoid excessive nitrogen, ensure proper spacing.",
    },
    "Rice___Neck_Blast": {
        "display_name": "Rice Neck Blast",
        "crop": "Rice",
        "status": "Diseased",
        "severity": "Very High",
        "description": "Also caused by Magnaporthe oryzae, but infects the neck node of the panicle — causes most severe yield loss.",
        "symptoms": ["Dark brown/black lesion at panicle neck", "Whitening and empty panicles (white ear)", "Lodging of infected panicles"],
        "treatment": "Apply tricyclazole at heading stage. Immediate fungicide application is critical.",
        "prevention": "Apply preventive fungicide at booting to heading stage. Use resistant varieties.",
    },
    "Sugarcane_Bacterial Blight": {
        "display_name": "Sugarcane Bacterial Blight",
        "crop": "Sugarcane",
        "status": "Diseased",
        "severity": "High",
        "description": "Caused by Xanthomonas albilineans. Spread through infected planting material and contaminated tools.",
        "symptoms": ["White to pale yellow leaf streaks", "Pencil-line streaks along veins", "Leaf scalding and wilting"],
        "treatment": "No effective chemical cure. Rogue infected stools. Use disease-free planting material.",
        "prevention": "Plant certified disease-free sets, disinfect cutting tools between rows.",
    },
    "Sugarcane_Healthy": {
        "display_name": "Healthy Sugarcane",
        "crop": "Sugarcane",
        "status": "Healthy",
        "severity": "None",
        "description": "The sugarcane plant appears healthy with no visible disease symptoms.",
        "symptoms": [],
        "treatment": "No treatment needed.",
        "prevention": "Use certified planting material, balanced nutrition, regular field monitoring.",
    },
    "Sugarcane_Red Rot": {
        "display_name": "Sugarcane Red Rot",
        "crop": "Sugarcane",
        "status": "Diseased",
        "severity": "High",
        "description": "Caused by Colletotrichum falcatum. Major disease of sugarcane affecting stalk internals.",
        "symptoms": ["Red discoloration inside stalk", "White patches alternating with red in stalk", "Leaves yellowing and drying"],
        "treatment": "Remove and destroy infected plants. Treat seeds with carbendazim fungicide.",
        "prevention": "Plant resistant varieties, use hot water treated setts, crop rotation.",
    },
    "Wheat___Brown_Rust": {
        "display_name": "Wheat Brown Rust",
        "crop": "Wheat",
        "status": "Diseased",
        "severity": "Moderate to High",
        "description": "Caused by Puccinia triticina. The most common rust disease of wheat, spread by airborne urediniospores.",
        "symptoms": ["Orange-brown pustules mainly on upper leaf surface", "Random scattered pustule distribution", "Premature leaf death"],
        "treatment": "Apply propiconazole or tebuconazole fungicides. Apply at early pustule stage.",
        "prevention": "Plant resistant varieties, early sowing, apply preventive fungicide.",
    },
    "Wheat___Healthy": {
        "display_name": "Healthy Wheat",
        "crop": "Wheat",
        "status": "Healthy",
        "severity": "None",
        "description": "The wheat plant appears healthy with no signs of rust or other diseases.",
        "symptoms": [],
        "treatment": "No treatment needed.",
        "prevention": "Monitor for early rust signs, balanced nutrition, use certified seed.",
    },
    "Wheat___Yellow_Rust": {
        "display_name": "Wheat Yellow Rust",
        "crop": "Wheat",
        "status": "Diseased",
        "severity": "High",
        "description": "Caused by Puccinia striiformis. Thrives in cool, moist weather. Can devastate entire crops if not managed early.",
        "symptoms": ["Yellow-orange pustules in stripes along leaf veins", "Powdery yellow spore masses", "Affected plants may die early"],
        "treatment": "Apply tebuconazole or propiconazole at first sign of infection. Repeat every 14 days if conditions persist.",
        "prevention": "Plant resistant varieties, early detection and spraying, avoid late sowing.",
    },
}


class CropDiseasePredictor:
    def __init__(self):
        self.model = None
        self.classes = None
        self.transform = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {METADATA_PATH}. "
                "Please run train.py first to train the model."
            )
        with open(METADATA_PATH) as f:
            meta = json.load(f)

        self.classes = meta["classes"]
        img_size = meta.get("img_size", 224)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, len(self.classes)),
        )
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
        model.eval()
        self.model = model
        self._loaded = True

    def predict(self, image: Image.Image) -> dict:
        self.load()
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_prob, top_idx = probs.topk(5)
        top5 = [
            {"class": self.classes[i], "probability": round(p * 100, 2)}
            for i, p in zip(top_idx.tolist(), top_prob.tolist())
        ]

        predicted_class = self.classes[top_idx[0].item()]
        confidence = round(top_prob[0].item() * 100, 2)
        info = DISEASE_INFO.get(predicted_class, {})

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top5": top5,
            "disease_info": info,
        }


# Singleton
predictor = CropDiseasePredictor()
