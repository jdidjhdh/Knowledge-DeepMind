CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_points (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact TEXT NOT NULL,
    category VARCHAR(50),
    confidence FLOAT DEFAULT 0.5,
    related_entities TEXT[],
    source TEXT,
    source_document_id VARCHAR(255),
    embedding vector(512),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    source_path VARCHAR(500),
    source_type VARCHAR(50),
    chunk_index INT DEFAULT 0,
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    embedding vector(512),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id VARCHAR(255),
    role VARCHAR(50),
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    preferences JSONB DEFAULT '{}',
    focused_topics TEXT[],
    interaction_history JSONB DEFAULT '[]',
    correction_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 分类系统核心表
CREATE TABLE IF NOT EXISTS categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parent_id UUID REFERENCES categories(id) ON DELETE SET NULL,
    category_type VARCHAR(50) DEFAULT 'structural',
    level INT DEFAULT 0,
    embedding vector(512),
    semantic_vector JSONB,
    knowledge_count INT DEFAULT 0,
    avg_confidence FLOAT DEFAULT 0.5,
    is_archived BOOLEAN DEFAULT FALSE,
    is_frozen BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_accessed_at TIMESTAMP DEFAULT NOW()
);

-- 知识点多对多关联分类
CREATE TABLE IF NOT EXISTS knowledge_category (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_id UUID NOT NULL REFERENCES knowledge_points(id) ON DELETE CASCADE,
    category_id UUID NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    confidence FLOAT DEFAULT 0.5,
    is_primary BOOLEAN DEFAULT FALSE,
    is_auto_assigned BOOLEAN DEFAULT TRUE,
    assigned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(knowledge_id, category_id)
);

-- 分类之间的关系 (DAG)
CREATE TABLE IF NOT EXISTS category_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_category_id UUID NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    target_category_id UUID NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_category_id, target_category_id, relation_type)
);

-- 用户自定义标签
CREATE TABLE IF NOT EXISTS user_tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    color VARCHAR(7) DEFAULT '#6366f1',
    dimension VARCHAR(50) DEFAULT 'custom',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, name)
);

-- 知识点与用户标签多对多
CREATE TABLE IF NOT EXISTS knowledge_tag (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_id UUID NOT NULL REFERENCES knowledge_points(id) ON DELETE CASCADE,
    tag_id UUID NOT NULL REFERENCES user_tags(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    assigned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(knowledge_id, tag_id, user_id)
);

-- 智能集合定义
CREATE TABLE IF NOT EXISTS smart_collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    filter_rules JSONB NOT NULL DEFAULT '{}',
    is_system BOOLEAN DEFAULT FALSE,
    sort_order INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 用户分类偏好
CREATE TABLE IF NOT EXISTS user_category_prefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    category_id UUID NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    is_visible BOOLEAN DEFAULT TRUE,
    is_expanded BOOLEAN DEFAULT FALSE,
    sort_order INT DEFAULT 0,
    custom_name VARCHAR(255),
    visit_count INT DEFAULT 0,
    last_visited_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, category_id)
);

-- 分类进化历史
CREATE TABLE IF NOT EXISTS category_evolution_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category_id UUID REFERENCES categories(id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}',
    triggered_by VARCHAR(50) DEFAULT 'system',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kp_embedding ON knowledge_points USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_doc_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_conv_id ON conversation_history(conversation_id);

CREATE INDEX IF NOT EXISTS idx_cat_parent ON categories(parent_id);
CREATE INDEX IF NOT EXISTS idx_cat_type ON categories(category_type);
CREATE INDEX IF NOT EXISTS idx_cat_archived ON categories(is_archived);
CREATE INDEX IF NOT EXISTS idx_kc_knowledge ON knowledge_category(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_kc_category ON knowledge_category(category_id);
CREATE INDEX IF NOT EXISTS idx_cr_source ON category_relations(source_category_id);
CREATE INDEX IF NOT EXISTS idx_cr_target ON category_relations(target_category_id);
CREATE INDEX IF NOT EXISTS idx_kt_knowledge ON knowledge_tag(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_kt_tag ON knowledge_tag(tag_id);
CREATE INDEX IF NOT EXISTS idx_kt_user ON knowledge_tag(user_id);
CREATE INDEX IF NOT EXISTS idx_ucp_user ON user_category_prefs(user_id);
CREATE INDEX IF NOT EXISTS idx_ucp_category ON user_category_prefs(category_id);

CREATE INDEX IF NOT EXISTS idx_kp_created_id ON knowledge_points(created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_kp_category_created ON knowledge_points(category, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_kp_confidence_created ON knowledge_points(confidence, created_at DESC);

CREATE OR REPLACE FUNCTION search_knowledge_points(
    query_embedding vector(512),
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE(
    id UUID,
    fact TEXT,
    category VARCHAR,
    confidence FLOAT,
    source TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kp.id,
        kp.fact,
        kp.category,
        kp.confidence,
        kp.source,
        1 - (kp.embedding <=> query_embedding) AS similarity
    FROM knowledge_points kp
    WHERE 1 - (kp.embedding <=> query_embedding) > match_threshold
    ORDER BY kp.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;