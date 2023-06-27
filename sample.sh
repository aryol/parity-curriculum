for task in parity5 parity7 parity10
do
    for samples in 10000 31616 100000 316224 1000000
    do
        for seed in {1..10}
        do
            python3 main.py -task $task -model mlp -lr 0.003 -seed $seed -p 0.01 -rho 0.01 -curr 1 -train-size $samples -compute-int 10000 -cuda 0
        done
        wait
    done
done

# one can use sample size of 0 for the fresh samples
