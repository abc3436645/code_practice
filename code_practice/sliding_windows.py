from collections import defaultdict

def min_window(s:str, t:str):
    need = defaultdict(int)
    window = defaultdict(int)

    for c in t:
        need[c] += 1
    
    left , right = 0, 0 
    valid = 0 
    while right < len(s):
        c = s[right]
        right += 1
        
        if c in need:
            window[c] += 1
            if window[c] == need[c]:
                vaild += 1
        while valid  == len(need):
            if right - left < 0: # TODO
                pass
