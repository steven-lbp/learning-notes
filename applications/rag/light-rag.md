# LightRAG

## Why LightRAG
- Graph-Based Text Indexing: Transforms textual data into a knowledge graph
- Dual-Level Retrieval System: 
Low-Level Retrieval and High-Level Retrieval
- Incremental Update Algorithm: 
Allows real-time integration of new data into the knowledge graph

## Overall Architecture
1. 
![image-20250201124929335](images/1.png)

`D(.)` : Deduplication

`P(.)` : Profiling

`R(.)` : Relationship Extraction

2. 
![image-20250201124929335](images/2.png)

Entities and relationships are stored in a graph structure for fast and semantically rich retrieval.

Nodes = concepts/entities, edges = relations

3. 
![image-20250201124929335](images/3.png)

Given a **user query**, the system retrieves both:

- **Low-Level Keys**: Directly mentioned terms (e.g., Beekeeper, Honey Bee, Hive).

- **High-Level Keys**: LLM-generated abstract concepts (e.g., Agriculture, Production, Environmental Impact).
