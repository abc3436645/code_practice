# 选择排序
"""
「选择排序 selection sort」的工作原理非常直接：开启一个循环，每轮从未排序区间选择最小的元素，将其
放到已排序区间的末尾。
设数组的长度为 𝑛 ，选择排序的算法流程如图 11‑2 所示。
1. 初始状态下，所有元素未排序，即未排序（索引）区间为 [0, 𝑛 − 1] 。
2. 选取区间 [0, 𝑛 − 1] 中的最小元素，将其与索引 0 处元素交换。完成后，数组前 1 个元素已排序。
3. 选取区间 [1, 𝑛 − 1] 中的最小元素，将其与索引 1 处元素交换。完成后，数组前 2 个元素已排序。
4. 以此类推。经过 𝑛 − 1 轮选择与交换后，数组前 𝑛 − 1 个元素已排序。
5. 仅剩的一个元素必定是最大元素，无须排序，因此数组排序完成。
"""
from typing import List

def selection_sort(nums:List[int]):
    """选择排序：每次循环将最小的元素放在已经排序的数组的末尾

    Args:
        nums (List[int]): _description_
    """
    n = len(nums)
    # 外层循环未排序区间[i,n-1]
    for i in range(n-1): # 
        k = i 
        for j in range(i+1,n): 
            # 内层循环找到未排序的最小元素
            if nums[j] < nums[k]:
                k = j   # 找到最小元素索引
        nums[i], nums[k] = nums[k], nums[i] # 将最小元素和未排序的数组的首个元素进行交换
    
    return nums

# 冒泡排序
"""
通过连续地比较与交换相邻元素实现排序。这个过程就像气泡从底部升到顶部一样，
因此得名冒泡排序。
"""
def bubble_sort(nums:List[int]):
    """冒泡排序
    1. 首先，对 𝑛 个元素执行“冒泡”，将数组的最大元素交换至正确位置，
    2. 接下来，对剩余 𝑛 − 1 个元素执行“冒泡”，将第二大元素交换至正确位置。
    3. 以此类推，经过 𝑛 − 1 轮“冒泡”后，前 𝑛 − 1 大的元素都被交换至正确位置。
    4. 仅剩的一个元素必定是最小元素，无须排序，因此数组排序完成

    优化点：如果内层循环没有发生交换，说明已经有序，直接跳出当前循环
    Args:
        nums (List[int]): _description_

    Returns:
        _type_: _description_
    """
    n = len(nums)
    for i in range(n-1,0,-1): # 控制外层循环数
        flag = False
        for j in range(i): # 利用i进行内层循环比较
            if nums[j] > nums[j+1]:
                flag = True
                nums[j],nums[j+1] = nums[j+1],nums[j] 
        if not flag:
            break
    return nums

# 插入排序
"""插入排序
1. 初始状态下，数组的第 1 个元素已完成排序。
2. 选取数组的第 2 个元素作为 base ，将其插入到正确位置后，数组的前 2 个元素已排序。
3. 选取第 3 个元素作为 base ，将其插入到正确位置后，数组的前 3 个元素已排序。
4. 以此类推，在最后一轮中，选取最后一个元素作为 base ，将其插入到正确位置后，所有元素均已排序。
"""
def insert_sort(nums:List[int]):
    """ 插入排序"""
    # 外循环：已排序区间为 [0, i-1]
    for i in range(1, len(nums)):
        base = nums[i]
        j = i - 1
        # 内循环：将 base 插入到已排序区间 [0, i-1] 中的正确位置
        while j >= 0 and nums[j] > base:
            nums[j + 1] = nums[j] # 将 nums[j] 向右移动一位
            j -= 1
        nums[j + 1] = base # 将 base 赋值到正确位置

    return nums

# 快速排序：哨兵划分的实质是将一个较长数组的排序问题简化为两个较短数组的排序问题。
def partition(nums:List[int],left:int,right:int):
    """ 哨兵划分"""
    # 以 nums[left] 作为基准数
    i, j = left, right
    while i < j:
        while i < j and nums[j] >= nums[left]:
            j -= 1 # 从右向左找首个小于基准数的元素
        while i < j and nums[i] <= nums[left]:
            i += 1 # 从左向右找首个大于基准数的元素
        # 元素交换
        nums[i], nums[j] = nums[j], nums[i]
    # 将基准数交换至两子数组的分界线
    nums[i], nums[left] = nums[left], nums[i]
    return i # 返回基准数的索引

def quick_sort(nums:List[int],left:int,right:int):
    """ 快速排序"""
    # 子数组长度为 1 时终止递归
    if left >= right:
        return
    # 哨兵划分
    pivot = partition(nums, left, right)
    # 递归左子数组、右子数组
    quick_sort(nums, left, pivot - 1)
    quick_sort(nums, pivot + 1, right)

    return nums


"""归并排序:是一种基于分治策略的排序算法,包含“划分”和“合并”阶段。
1. 划分阶段：通过递归不断地将数组从中点处分开，将长数组的排序问题转换为短数组的排序问题。
2. 合并阶段：当子数组长度为 1 时终止划分，开始合并，持续地将左右两个较短的有序数组合并为一个较
长的有序数组，直至结束。

1. 计算数组中点 mid ，递归划分左子数组（区间 [left, mid] ）和右子数组（区间 [mid + 1, right] ）。
2. 递归执行步骤 1. ，直至子数组区间长度为 1 时，终止递归划分。
“合并阶段”从底至顶地将左子数组和右子数组合并为一个有序数组。需要注意的是，从长度为 1 的子数组开
始合并，合并阶段中的每个子数组都是有序的。
"""
def merge(nums: List[int], left: int, mid: int, right: int):
    """ 合并左子数组和右子数组"""
    # 左子数组区间 [left, mid], 右子数组区间 [mid+1, right]
    # 创建一个临时数组 tmp ，用于存放合并后的结果
    tmp = [0] * (right - left + 1)
    # 初始化左子数组和右子数组的起始索引
    i, j, k = left, mid + 1, 0
    # 当左右子数组都还有元素时，比较并将较小的元素复制到临时数组中
    while i <= mid and j <= right:
        if nums[i] <= nums[j]:
            tmp[k] = nums[i]
            i += 1
        else:
            tmp[k] = nums[j]
            j += 1
        k += 1
    # 将左子数组和右子数组的剩余元素复制到临时数组中
    while i <= mid:
        tmp[k] = nums[i]
        i += 1
        k += 1
    while j <= right:
        tmp[k] = nums[j]
        j += 1
        k += 1
    # 将临时数组 tmp 中的元素复制回原数组 nums 的对应区间
    for k in range(0, len(tmp)):
        nums[left + k] = tmp[k]

def merge_sort(nums: List[int], left: int, right: int):
    """ 归并排序"""
    # 终止条件
    if left >= right:
        return # 当子数组长度为 1 时终止递归
    # 划分阶段
    mid = (left + right) // 2 # 计算中点
    merge_sort(nums, left, mid) # 递归左子数组
    merge_sort(nums, mid + 1, right) # 递归右子数组
    # 合并阶段
    merge(nums, left, mid, right)
    return nums

if __name__ == "__main__":
    nums = [1,3,2,1,2,3,4,5]
    print(selection_sort(nums))
    print(bubble_sort(nums))
    print(insert_sort(nums))
    print(merge_sort(nums,0,len(nums)-1))
    print(quick_sort(nums,0,len(nums)-1))