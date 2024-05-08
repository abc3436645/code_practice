###209长度最小的数组
## 滑动窗口解法

def min_sub_array(s:int,nums:list):
    """
    暴力法
    """
    res = float('inf')
    sublength = 0
    for i in range(len(nums)):
        windows_sum = 0
        for j in range(i,len(nums)):
            windows_sum += nums[j]
            if windows_sum >= s:
                sublength = j-i+1
                if res < sublength:
                    res = sublength
                break
    return 0 if res == float('inf') else res


def min_sub_array(s:int,nums:list):
    """滑动窗口"""
    res = float('inf')
    windows_sum = 0
    sublength = 0
    i = 0
    for j in range(len(nums)):
        windows_sum += nums[j]
        while windows_sum >= s:
            sublength = j - i + 1
            if res > sublength:
                res = sublength
                windows_sum -= nums[i]
                i += 1
    return 0 if res == float('inf') else res

