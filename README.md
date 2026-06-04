# Executing the code:

```bash
python main.py $n $r $b $s $seed $nr_simul
```

# Arguments:
* $n = number of teams (18 or 36)
* $r =  number of rounds (if 18 teams: 6 or 9, if 36 teams: 8 or 14)
* $b = prize distribution (0 (large middle),1 (evenly distributed), 2 (small middle))
* $s = strength distribution (0 (uniform), 1 (left-tailed), 2 (right-tailed), 3 (strong center), 4 (empirical), 5 (all same strength))
* $seed = seed 
* $nr_simul = number of simulations (max 5000)