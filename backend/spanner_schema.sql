-- Semantic Node Table
CREATE TABLE SemanticClipNode (
    node_id STRING(MAX) NOT NULL,
    video_uri STRING(MAX),
    start_time_ms INT64,
    end_time_ms INT64,
    transcript_text STRING(MAX),
    vocal_delivery STRING(MAX),
    speakers JSON,
    objects_present JSON,
    visual_labels JSON,
    content_mechanisms JSON,
    embedding ARRAY<FLOAT32>(vector_length=>3072),
    spatial_tracking_uri STRING(MAX)
) PRIMARY KEY (node_id);

-- Narrative Edge Table
CREATE TABLE NarrativeEdge (
    edge_id STRING(MAX) NOT NULL,
    from_node_id STRING(MAX) NOT NULL,
    to_node_id STRING(MAX) NOT NULL,
    label STRING(MAX),
    narrative_classification STRING(MAX),
    confidence_score FLOAT64
) PRIMARY KEY (edge_id);

-- Native Vector Index (ScaNN)
CREATE VECTOR INDEX ClyptSemanticIndex 
ON SemanticClipNode(embedding) 
WHERE embedding IS NOT NULL 
OPTIONS (distance_type = 'COSINE', tree_depth = 2, num_leaves = 1000);

-- Bind the Tables into the Property Graph
CREATE PROPERTY GRAPH ClyptGraph
  NODE TABLES (SemanticClipNode)
  EDGE TABLES (NarrativeEdge
    SOURCE KEY (from_node_id) REFERENCES SemanticClipNode(node_id)
    DESTINATION KEY (to_node_id) REFERENCES SemanticClipNode(node_id)
  );
