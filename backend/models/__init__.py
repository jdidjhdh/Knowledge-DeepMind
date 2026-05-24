from pydantic import BaseModel, Field
from typing import Optional, Union
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    WORD = "word"
    PPT = "ppt"
    IMAGE = "image"
    WEB = "web"
    TABLE = "table"
    CODE = "code"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class FactCategory(str, Enum):
    CONCEPT = "概念"
    FACT = "事实"
    METHOD = "方法"
    OPINION = "观点"
    PENDING = "待验证"


class CategoryType(str, Enum):
    STRUCTURAL = "structural"
    META = "meta"
    DYNAMIC_VIEW = "dynamic_view"
    SMART_COLLECTION = "smart_collection"
    TEMPORAL = "temporal"
    SOURCE = "source"


class CategoryRelationType(str, Enum):
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    RELATED_TO = "RELATED_TO"
    EQUIVALENT_TO = "EQUIVALENT_TO"
    CONTRADICTS = "CONTRADICTS"
    EXPANDS = "EXPANDS"


class EvolutionAction(str, Enum):
    SPLIT = "split"
    MERGE = "merge"
    CREATE = "create"
    ARCHIVE = "archive"
    REASSIGN = "reassign"
    FREEZE = "freeze"


class ConfidenceTier(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    DOUBTFUL = "doubtful"


class EntityType(str, Enum):
    ENTITY = "Entity"
    CONCEPT = "Concept"
    EVENT = "Event"
    DOCUMENT = "Document"
    KNOWLEDGE_ATOM = "KnowledgeAtom"
    FUNCTION = "Function"
    CLASS = "Class"
    MODULE = "Module"
    ALGORITHM = "Algorithm"
    FILE = "File"


class RelationType(str, Enum):
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    INSTANCE_OF = "INSTANCE_OF"
    CAUSES = "CAUSES"
    DEPENDS_ON = "DEPENDS_ON"
    INDICATES = "INDICATES"
    BELONGS_TO = "BELONGS_TO"
    OCCURS_AT = "OCCURS_AT"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    EVIDENCED_BY = "EVIDENCED_BY"
    CONFIRMED_BY = "CONFIRMED_BY"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    ENDORSED_BY = "ENDORSED_BY"
    REVISED_TO = "REVISED_TO"
    RELATED_TO = "RELATED_TO"
    DEFINES_FUNCTION = "DEFINES_FUNCTION"
    DEFINES_CLASS = "DEFINES_CLASS"
    CALLS = "CALLS"
    IMPLEMENTS_ALGORITHM = "IMPLEMENTS_ALGORITHM"
    IMPORTS = "IMPORTS"
    RELATED_TO_CONCEPT = "RELATED_TO_CONCEPT"
    BELONGS_TO_MODULE = "BELONGS_TO_MODULE"


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    content: str
    source_path: str
    source_type: DocumentType
    chunk_index: int = 0
    timestamp: Optional[str] = None
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)


class KnowledgePoint(BaseModel):
    id: Optional[str] = None
    fact: str
    category: FactCategory
    confidence: float = Field(ge=0.0, le=1.0)
    related_entities: list[str] = Field(default_factory=list)
    source: str
    source_document_id: Optional[str] = None
    created_at: Optional[datetime] = None
    model_confidence_raw: float = Field(ge=0.0, le=1.0, default=0.5)
    source_quality: float = Field(ge=0.0, le=1.0, default=0.5)
    consistency_score: float = Field(ge=0.0, le=1.0, default=0.5)
    feedback_alpha: float = Field(default=2.0)
    feedback_beta: float = Field(default=2.0)
    calibrated_confidence: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    last_updated: Optional[datetime] = None
    interaction_count: int = 0
    status: str = "active"
    replaced_by: Optional[str] = None
    history: list[dict] = Field(default_factory=list)
    event_time: Optional[str] = None
    time_precision: Optional[str] = None
    event_times: list[dict] = Field(default_factory=list)
    version: int = 1
    is_active: bool = True


class KnowledgeTriple(BaseModel):
    subject: str
    relation: str
    object: str
    relation_type: RelationType = RelationType.RELATED_TO
    source_knowledge_id: Optional[str] = None
    source_chunk_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class GraphNodeDetail(BaseModel):
    id: str
    label: str
    node_type: EntityType = EntityType.ENTITY
    aliases: list[str] = Field(default_factory=list)
    canonical_name: Optional[str] = None
    confidence: float = 0.5
    degree: int = 0
    community_id: Optional[int] = None
    category_ids: list[str] = Field(default_factory=list)
    event_time: Optional[str] = None
    source_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class GraphEdgeDetail(BaseModel):
    source: str
    target: str
    relation: str
    relation_type: RelationType = RelationType.RELATED_TO
    confidence: float = 0.5
    source_knowledge_id: Optional[str] = None
    evidence_snippet: Optional[str] = None


class MultiHopPath(BaseModel):
    paths: list[list[str]]
    relations: list[list[str]]
    path_weights: list[float]
    source_entity: str
    target_entity: str
    max_hops: int


class CommunityResult(BaseModel):
    community_count: int
    modularity: Optional[float] = None
    communities: dict[int, list[str]] = Field(default_factory=dict)
    community_labels: dict[int, str] = Field(default_factory=dict)
    community_sizes: dict[int, int] = Field(default_factory=dict)


class EntityNormalizeRequest(BaseModel):
    entity_name: str
    entity_type: EntityType = EntityType.ENTITY
    force_merge: bool = False


class RuleInference(BaseModel):
    rule_name: str
    rule_description: str
    inferred_count: int
    new_triples: list[KnowledgeTriple] = Field(default_factory=list)


class GraphSyncRequest(BaseModel):
    enable_normalization: bool = True
    enable_evidence_chain: bool = True
    chunk_id: Optional[str] = None


class KnowledgeFeedback(BaseModel):
    knowledge_id: str
    feedback_type: str
    comment: Optional[str] = None


class KnowledgeCorrectRequest(BaseModel):
    fact: str
    category: Optional[str] = None
    source: Optional[str] = None


class LowConfidenceReviewRequest(BaseModel):
    threshold: float = 0.4
    limit: int = 10
    enable_external_search: bool = False


class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class ConversationRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    stream: bool = True
    enable_web_search: bool = True


class ConversationResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[dict] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)
    detected_conflicts: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)
    low_confidence_info: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    search_type: str = "hybrid"


class SearchResult(BaseModel):
    knowledge_points: list[KnowledgePoint]
    document_chunks: list[DocumentChunk]
    graph_results: list[dict] = Field(default_factory=list)


class IngestionTask(BaseModel):
    task_id: str
    file_path: str
    file_type: DocumentType
    status: str = "pending"
    progress: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None


class UserProfile(BaseModel):
    id: Optional[str] = None
    preferences: dict = Field(default_factory=dict)
    focused_topics: list[str] = Field(default_factory=list)
    interaction_history: list[dict] = Field(default_factory=list)
    correction_count: int = 0


class Category(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None
    category_type: CategoryType = CategoryType.STRUCTURAL
    level: int = 0
    icon: Optional[str] = None
    color: str = "#6366f1"
    sort_order: int = 0
    semantic_vector: Optional[list[float]] = None
    knowledge_count: int = 0
    avg_confidence: float = 0.5
    is_archived: bool = False
    is_frozen: bool = False
    is_system: bool = False
    metadata: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None


class KnowledgeCategoryAssignment(BaseModel):
    knowledge_id: str
    category_id: str
    confidence: float = 0.5
    is_primary: bool = False
    is_auto_assigned: bool = True
    assigned_at: Optional[datetime] = None


class CategoryRelation(BaseModel):
    source_category_id: str
    target_category_id: str
    relation_type: CategoryRelationType
    weight: float = 1.0
    metadata: dict = Field(default_factory=dict)


class UserTag(BaseModel):
    id: Optional[str] = None
    user_id: str
    name: str
    color: str = "#6366f1"
    dimension: str = "custom"


class KnowledgeTagAssignment(BaseModel):
    knowledge_id: str
    tag_id: str
    user_id: str


class SmartCollection(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    filter_rules: dict = Field(default_factory=dict)
    is_system: bool = False
    sort_order: int = 0


class UserCategoryPrefs(BaseModel):
    user_id: str
    category_id: str
    is_visible: bool = True
    is_expanded: bool = False
    sort_order: int = 0
    custom_name: Optional[str] = None
    visit_count: int = 0
    last_visited_at: Optional[datetime] = None


class CategoryTree(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    category_type: CategoryType
    level: int
    icon: Optional[str] = None
    color: str = "#6366f1"
    knowledge_count: int
    avg_confidence: float
    is_archived: bool
    is_frozen: bool
    is_system: bool = False
    children: list["CategoryTree"] = Field(default_factory=list)
    relations: list[CategoryRelation] = Field(default_factory=list)
    tags: list[UserTag] = Field(default_factory=list)


class CategoryHealth(BaseModel):
    category_id: str
    name: str
    knowledge_count: int
    avg_confidence: float
    last_updated: Optional[datetime] = None
    is_stale: bool = False
    needs_split: bool = False
    needs_attention: bool = False


class MultiDimensionFilter(BaseModel):
    category_ids: list[str] = Field(default_factory=list)
    tag_ids: list[str] = Field(default_factory=list)
    confidence_tier: Optional[ConfidenceTier] = None
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    source_type: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    event_time_from: Optional[str] = None
    event_time_to: Optional[str] = None
    search_text: Optional[str] = None
    status: Optional[str] = None
    sort_by: str = "created_at"
    sort_order: str = "desc"
    offset: int = 0
    limit: int = 50


class CategoryEvolutionEvent(BaseModel):
    category_id: Optional[str] = None
    action: EvolutionAction
    details: dict = Field(default_factory=dict)
    triggered_by: str = "system"
    created_at: Optional[datetime] = None


class ClusteringResult(BaseModel):
    num_clusters: int
    labels: list[int]
    noise_count: int
    cluster_keywords: dict[int, list[str]]
    cluster_sizes: dict[int, int]
    suggested_names: dict[int, str]


class KnowledgeTimeline(BaseModel):
    year: int
    month: Optional[int] = None
    day: Optional[int] = None
    knowledge_count: int
    event_label: Optional[str] = None
    knowledge_ids: list[str] = Field(default_factory=list)
    granularity: str = "month"


class TimelineGroup(BaseModel):
    time_key: str
    label: str
    record_count: int
    event_count: int
    items: list[dict] = Field(default_factory=list)
    is_gap: bool = False
    is_burst: bool = False
    confidence_avg: float = 0.0


class TimelineGap(BaseModel):
    start_date: str
    end_date: str
    duration_days: int
    label: str
    suggestion: str = ""


class TimelineBurst(BaseModel):
    center_date: str
    density_multiplier: float
    knowledge_count: int
    top_categories: list[str] = Field(default_factory=list)
    label: str = ""


class VersionChain(BaseModel):
    entity_name: str
    versions: list[dict] = Field(default_factory=list)
    latest_version: Optional[str] = None
    total_updates: int = 0


class TimelineResponse(BaseModel):
    groups: list[TimelineGroup] = Field(default_factory=list)
    total: int = 0
    mode: str = "event_time"
    granularity: str = "month"
    gaps: list[TimelineGap] = Field(default_factory=list)
    bursts: list[TimelineBurst] = Field(default_factory=list)
    version_chains: list[VersionChain] = Field(default_factory=list)


class PaginationResponse(BaseModel):
    mode: str = "offset"
    page: int = 1
    page_size: int = 20
    total: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_prev: bool = False
    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None


class KnowledgeListResponse(BaseModel):
    data: list[dict] = Field(default_factory=list)
    pagination: PaginationResponse = Field(default_factory=PaginationResponse)


class SourceGroup(BaseModel):
    source_name: str
    source_type: str
    knowledge_count: int
    avg_confidence: float
    knowledge_ids: list[str] = Field(default_factory=list)
    first_added: Optional[datetime] = None
    last_added: Optional[datetime] = None


class SourceComparison(BaseModel):
    topic: str
    source_a: SourceGroup
    source_b: SourceGroup
    conflicting_points: list[dict] = Field(default_factory=list)
    agreement_points: list[dict] = Field(default_factory=list)


class DedupMode(str, Enum):
    STRICT = "strict"
    LOOSE = "loose"
    MANUAL = "manual"


class FileHashRecord(BaseModel):
    file_hash: str
    file_name: str
    file_size: int
    file_type: DocumentType
    saved_path: str
    task_id: str
    content_text: str = ""
    content_vector: Optional[list[float]] = None
    created_at: Optional[datetime] = None


class DedupCheckRequest(BaseModel):
    file_name: str
    file_size: int
    file_content: Optional[str] = None


class DedupCheckResult(BaseModel):
    hash_match: bool = False
    hash_matched_file: Optional[FileHashRecord] = None
    content_similarity: float = 0.0
    similar_files: list[dict] = Field(default_factory=list)
    suggested_action: str = "proceed"
    message: str = ""
    is_duplicate: bool = False


class CategoryMergeSuggestion(BaseModel):
    category_a: Category
    category_b: Category
    similarity: float
    overlapping_knowledge: list[str] = Field(default_factory=list)
    suggestion: str = ""


class BatchCategoryAssignRequest(BaseModel):
    knowledge_ids: list[str]
    category_ids: list[str]
    primary_category_id: Optional[str] = None
    is_auto: bool = False


class AutoCategorizeResult(BaseModel):
    knowledge_id: str
    assigned_categories: list[str] = Field(default_factory=list)
    suggested_new: Optional[str] = None
    confidence: float = 0.0


class DedupStats(BaseModel):
    total_files_tracked: int = 0
    duplicates_found: int = 0
    strict_skipped: int = 0
    hash_store_size: int = 0