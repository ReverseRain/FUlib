-训练格式：python mufc_main.py --num_clusters=4(自定) --num_clients=100（自定） --data_path=celltype_processed.pkl（自定） --num_removes=10 (自定)           --k_prime=4  --split=non-iid  --compare_kfed --compare_dc --client_kpp_only --verbose --update_centralized_loss

-原项目只包含了celltype_processed.pkl数据集，剩下的数据在readme.md中有打包的链接，拆解出来的数据集格式不一致
-进行了简单的格式修改后，原数据存放在kmeans_data_deletion_NeurIPS19_datasets_scaled.p，拆包后的数据存放在data.zip，格式转化后的数据在根目录