Usage:
- Build
    - Copy the `environment.yml` file from `eimspred` root directory to `rassp/environment.yml`.
    - From the `rassp` directory, run `docker build -t simpleforward -f docker/Dockerfile .`
      This command expects `rassp/environment.yml` to exist, which we should have created/copied in the last step.
- Run inference
    - Put your parquet files for running inference against in a folder, and make sure they're world-readable via `chmod -R 777 parquet-data-folder/`.
    - Run the following command:
    ```
    docker run --shm-size=32g --cpus="32.0" --ulimit memlock=-1 -it --rm --network=host \
    -v /data/richardzhu/data/data.parquets/:/home/simpleforward/data.processed/ \
    simpleforward --filename="/home/simpleforward/data.processed/nist17_hconfspcl_48a_512_uf4096-all.sparse.parquet" \
    --num_data_workers=16 \
    --no-cuda \
    --format=parquet \
    --model_meta_filename="./models/subsetnet_best_candidate_nist_posttrain_cluster.cluster-nist-posttrain.35946017.meta" \
    --model_checkpoint_filename="./models/subsetnet_best_candidate_nist_posttrain_cluster.cluster-nist-posttrain.35946017.00000480.model"
    ```

a few lessons from that experience:
use mamba
pytorch should be a fixed version, e.g. 1.13 bc the API can and will change for important functions
some of the JIT stuff (eg numba) and tinygraph! requires a specific numpy version, so this should also be pinned
