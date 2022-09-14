def most_frequent(filepath):
    with open(filepath) as f:
        text = f.read()
        spl = text.lower().split()
        set_spl = list(set(spl))

    wc ={}
    
    for set_i in set_spl:
        
            count = 0
            
            for spl_i in spl:

                if set_i == spl_i:
                    count = count + 1

            wc[set_i] = count
            
    li_sorted = sorted(wc, key=lambda x: (-(wc[x]),x))
    sliced_sorted = li_sorted[0:9]
    return sliced_sorted

