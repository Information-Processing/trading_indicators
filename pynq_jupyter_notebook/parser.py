with open('main1k.txt', 'r') as file:
    content = file.read().splitlines()
    sw_unopt_total = 0.0
    sw_opt_total = 0.0
    hw_total = 0.0
    sample_count = 0
    count = 0
    for line in content:
        if "Samples" in line:
            words = line.split(" ")
            if float(words[12][0:6]) > float(words[9][0:6]):
                print(line)
                continue
            sample_count += int(words[1])
            sw_unopt_total += float(words[5][0:6])
            sw_opt_total += float(words[9][0:6])
            hw_total += float(words[12][0:6])
            count += 1
            #print(line)
    
    sw_unopt_avg = sw_unopt_total / count
    sw_opt_avg = sw_opt_total / count
    hw_avg = hw_total / count

    print(f"{sw_unopt_avg=}, {sw_opt_avg=}, {hw_avg=}, {count=}")
    #print(content)