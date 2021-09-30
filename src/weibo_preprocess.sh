#you need to get the raw dataset first
echo 'initializing...'
python src/preprocess/construct_graph/weibo_graph/adj_feature.py
#VGAE
echo 'getting graph embedding...'
python src/preprocess/GAE_VGAE/VGAE.py src/preprocess/prepare_data/ src/preprocess/prepare_data/weibo_features.txt src/preprocess/prepare_data/adj_matrix.txt

echo 'little modification...'
cp src/preprocess/prepare_data/vertex.txt src/preprocess/prepare_data/vertex_temp.txt
python src/preprocess/prepare_data/modify_weibo.py
rm src/preprocess/prepare_data/vertex_temp.txt
rm src/preprocess/prepare_data/graph_embedding_temp.npy
rm src/preprocess/prepare_data/weibo_features.txt

mv src/preprocess/prepare_data/vertex.txt graph_data/weibo/.
mv src/preprocess/prepare_data/adj_matrix.txt graph_data/weibo/.
mv src/preprocess/prepare_data/weibo_graph_embedding.npy graph_data/weibo/graph_embedding.npy
mv resource/weibo raw_data/weibo

