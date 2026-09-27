from enum import Enum


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class EmotionLabel(str, Enum):
    ANGER = "anger"
    FEAR = "fear"
    CONCERN = "concern"
    DISAPPOINTMENT = "disappointment"
    EXCITEMENT = "excitement"
    APPROVAL = "approval"
    CONFUSION = "confusion"
    SKEPTICISM = "skepticism"
    FRUSTRATION = "frustration"
    NEUTRAL = "neutral"


class ConcernCategory(str, Enum):
    PRIVACY = "privacy"
    SECURITY = "security"
    DATA_COLLECTION = "data_collection"
    COPYRIGHT = "copyright"
    BIAS = "bias"
    SAFETY = "safety"
    ACCURACY = "accuracy"
    RELIABILITY = "reliability"
    COST = "cost"
    PERFORMANCE = "performance"
    JOB_DISPLACEMENT = "job_displacement"
    SKILL_EROSION = "skill_erosion"
    USER_CONTROL = "user_control"
    TRANSPARENCY = "transparency"
    VENDOR_LOCK_IN = "vendor_lock_in"
    CODE_QUALITY = "code_quality"
    MISINFORMATION = "misinformation"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    ACCESSIBILITY = "accessibility"
    OTHER = "other"


class EntityType(str, Enum):
    ORG = "ORG"
    PRODUCT = "PRODUCT"
    PERSON = "PERSON"
    TECHNOLOGY = "TECHNOLOGY"
    EVENT = "EVENT"
    OTHER = "OTHER"


EMOTION_TAXONOMY: list[EmotionLabel] = [
    EmotionLabel.ANGER,
    EmotionLabel.FEAR,
    EmotionLabel.CONCERN,
    EmotionLabel.DISAPPOINTMENT,
    EmotionLabel.EXCITEMENT,
    EmotionLabel.APPROVAL,
    EmotionLabel.CONFUSION,
    EmotionLabel.SKEPTICISM,
    EmotionLabel.FRUSTRATION,
    EmotionLabel.NEUTRAL,
]

CONCERN_TAXONOMY: list[ConcernCategory] = [
    ConcernCategory.PRIVACY,
    ConcernCategory.SECURITY,
    ConcernCategory.DATA_COLLECTION,
    ConcernCategory.COPYRIGHT,
    ConcernCategory.BIAS,
    ConcernCategory.SAFETY,
    ConcernCategory.ACCURACY,
    ConcernCategory.RELIABILITY,
    ConcernCategory.COST,
    ConcernCategory.PERFORMANCE,
    ConcernCategory.JOB_DISPLACEMENT,
    ConcernCategory.SKILL_EROSION,
    ConcernCategory.USER_CONTROL,
    ConcernCategory.TRANSPARENCY,
    ConcernCategory.VENDOR_LOCK_IN,
    ConcernCategory.CODE_QUALITY,
    ConcernCategory.MISINFORMATION,
    ConcernCategory.ENVIRONMENTAL_IMPACT,
    ConcernCategory.ACCESSIBILITY,
    ConcernCategory.OTHER,
]

ENTITY_TYPE_TAXONOMY: list[EntityType] = [
    EntityType.ORG,
    EntityType.PRODUCT,
    EntityType.PERSON,
    EntityType.TECHNOLOGY,
    EntityType.EVENT,
    EntityType.OTHER,
]

SENTIMENT_LABELS: list[SentimentLabel] = [
    SentimentLabel.POSITIVE,
    SentimentLabel.NEUTRAL,
    SentimentLabel.NEGATIVE,
]

CONCERN_KEYWORDS: dict[ConcernCategory, list[str]] = {
    ConcernCategory.PRIVACY: [
        "privacy", "personal data", "user data", "data privacy", "surveillance",
        "tracking", "monitoring", "data collection", "personal information",
        "sensitive information", "data protection", "gdpr", "consent",
    ],
    ConcernCategory.SECURITY: [
        "security", "vulnerability", "exploit", "breach", "hack", "attack",
        "malware", "ransomware", "cybersecurity", "threat", "insecure",
        "unauthorized access", "penetration", "compromise",
    ],
    ConcernCategory.DATA_COLLECTION: [
        "data collection", "data gathering", "telemetry", "analytics",
        "usage data", "behavioral data", "harvesting", "scraping",
        "data harvesting", "mass collection",
    ],
    ConcernCategory.COPYRIGHT: [
        "copyright", "intellectual property", "ip infringement", "plagiarism",
        "licensing", "fair use", "dmca", "attribution", "proprietary",
        "training data", "creative work",
    ],
    ConcernCategory.BIAS: [
        "bias", "discrimination", "prejudice", "unfair", "discriminatory",
        "algorithmic bias", "systemic bias", "representation", "fairness",
        "equity", "disparate impact",
    ],
    ConcernCategory.SAFETY: [
        "safety", "harm", "dangerous", "risk", "unsafe", "hazard",
        "misuse", "abuse", "weaponize", "guardrail", "alignment",
        "existential risk",
    ],
    ConcernCategory.ACCURACY: [
        "accuracy", "hallucination", "incorrect", "wrong", "error",
        "mistake", "false", "misinformation", "fabrication", "reliability",
        "truthfulness", "factual",
    ],
    ConcernCategory.RELIABILITY: [
        "reliability", "downtime", "outage", "failure", "crash", "bug",
        "unstable", "inconsistent", "unreliable", "availability",
        "uptime", "robustness",
    ],
    ConcernCategory.COST: [
        "cost", "expensive", "price", "pricing", "subscription", "fee",
        "affordable", "budget", "financial", "monetization", "paywall",
        "free tier", "enterprise pricing",
    ],
    ConcernCategory.PERFORMANCE: [
        "performance", "speed", "latency", "slow", "lag", "throughput",
        "efficiency", "resource usage", "memory", "cpu", "gpu",
        "optimization", "scalability",
    ],
    ConcernCategory.JOB_DISPLACEMENT: [
        "job loss", "unemployment", "displacement", "automation",
        "replace workers", "labor market", "job security", "redundancy",
        "workforce reduction", "layoff",
    ],
    ConcernCategory.SKILL_EROSION: [
        "skill erosion", "deskilling", "dependency", "overreliance",
        "loss of skill", "atrophy", "understanding", "knowledge loss",
        "learning", "education", "critical thinking",
    ],
    ConcernCategory.USER_CONTROL: [
        "user control", "agency", "autonomy", "consent", "opt-out",
        "control", "choice", "preference", "customization", "override",
        "human in the loop",
    ],
    ConcernCategory.TRANSPARENCY: [
        "transparency", "explainability", "black box", "opaque",
        "interpretable", "auditable", "accountable", "disclosure",
        "open source", "audit",
    ],
    ConcernCategory.VENDOR_LOCK_IN: [
        "vendor lock-in", "lock-in", "proprietary", "migration",
        "interoperability", "portability", "standard", "walled garden",
        "ecosystem", "dependency",
    ],
    ConcernCategory.CODE_QUALITY: [
        "code quality", "technical debt", "maintainability", "readability",
        "bugs", "vulnerabilities", "security flaws", "best practices",
        "refactoring", "architecture",
    ],
    ConcernCategory.MISINFORMATION: [
        "misinformation", "disinformation", "fake", "false information",
        "propaganda", "manipulation", "deepfake", "synthetic media",
        "verification", "fact-check",
    ],
    ConcernCategory.ENVIRONMENTAL_IMPACT: [
        "environmental", "carbon", "energy", "sustainability", "climate",
        "emissions", "green", "power consumption", "compute", "footprint",
    ],
    ConcernCategory.ACCESSIBILITY: [
        "accessibility", "a11y", "inclusive", "disability", "usable",
        "accessible", "barrier", "assistive", "screen reader",
    ],
}

def get_concern_keywords(category: ConcernCategory) -> list[str]:
    return CONCERN_KEYWORDS.get(category, [])