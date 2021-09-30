#you need to get the raw dataset first
echo 'getting dataset...'
python src/preprocess/persona_process/prepare_data.py
echo 'constructing graph...'
python src/preprocess/construct_graph/persona_graph/construction.py src/preprocess/prepare_data/persona_graph.txt
python src/preprocess/construct_graph/persona_graph/adj_feature.py

#VGAE
echo 'getting graph embedding...'
python src/preprocess/GAE_VGAE/VGAE.py src/preprocess/prepare_data/ src/preprocess/prepare_data/features.txt src/preprocess/prepare_data/adj_matrix.txt

echo 'preprocess persona...'
cp src/preprocess/prepare_data/vertex.txt src/preprocess/prepare_data/vertex_temp.txt
python src/preprocess/prepare_data/modify.py
rm src/preprocess/prepare_data/vertex_temp.txt
rm src/preprocess/prepare_data/graph_embedding_temp.npy

python src/preprocess/get_context.py

mv src/preprocess/prepare_data/vertex.txt graph_data/persona/.
mv src/preprocess/prepare_data/adj_matrix.txt graph_data/persona/.
mv src/preprocess/prepare_data/graph_embedding.npy graph_data/persona/.
mv tx_data raw_data/persona

rm src/preprocess/prepare_data/features.txt
rm src/preprocess/prepare_data/persona_graph.txt
rm src/preprocess/source_data.pk
