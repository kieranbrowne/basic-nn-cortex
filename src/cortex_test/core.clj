(ns cortex-test.core
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.optimize.adam :as adam]
            [cortex.optimize.sgd :as sgd]
            [cortex.util :as util]
            [cortex.experiment.train :as experiment-train]
            ))


;; XOR dataset
(def dataset
  [{:data [0 1] :label [1]}
   {:data [1 0] :label [1]}
   {:data [1 1] :label [0]}
   {:data [0 0] :label [0]}])

(def num-features (count (:data (first dataset))))

;; describe linear network
(def network-description
  [(layers/input num-features 1 1 :id :data)
   (layers/linear->tanh 4 :bias (repeatedly 4 rand))
   (layers/linear->tanh 1 :id :label)
   ])

;; run untrained network
(execute/run
  (network/linear-network network-description)
  (create-dataset))


;; train network
(let [network (network/linear-network network-description)]
  (experiment-train/train-n
   network ; network hashmap
   dataset ;train-ds
   dataset ;test-ds
   :optimizer (sgd/sgd :learning-rate 0.01)
   :batch-size 4
   ))


;; running the trained network
(->> (execute/run
       (util/read-nippy-file "trained-network.nippy")
       (create-dataset))
     )
