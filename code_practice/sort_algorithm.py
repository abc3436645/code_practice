

def bubble_sort(nums:list):
    """
    冒泡算法核心：每次把最大最小的结果放在最后面，每次都是相邻的2个数字进行比较
    
    时间复杂度：O(N*N)
    """
    nums_len = len(nums)
    for i in range(nums_len-1): ##n-1趟
        for j in range(nums_len-i-1): ##遍历
            if nums[j] > nums[j+1]: ##控制算法从大到小或者从小到大排序
                nums[j], nums[j+1] = nums[j+1], nums[j]

    return nums

def quickSort(nums, left, right):  # 这种写法的平均空间复杂度为 O(logn) 
    # 分区操作
    def partition(nums, left, right):
        pivot = nums[left]  # 基准值
        while left < right:
            while left < right and nums[right] >= pivot:
                right -= 1
            nums[left] = nums[right]  # 比基准小的交换到前面
            while left < right and nums[left] <= pivot:
                left += 1
            nums[right] = nums[left]  # 比基准大交换到后面
        nums[left] = pivot # 基准值的正确位置，也可以为 nums[right] = pivot
        return left  # 返回基准值的索引，也可以为 return right
    # 递归操作
    if left < right:
        pivotIndex = partition(nums, left, right)
        quickSort(nums, left, pivotIndex - 1)  # 左序列
        quickSort(nums, pivotIndex + 1, right) # 右序列
    return nums
        

if __name__ == "__main__":
    nums = [1,34,4,15,27,2,9,8,45,66,96,36,48,100]

    num_sorted = bubble_sort(nums)
    print(num_sorted)

    num_quicksort = quickSort(nums,0,len(nums)-1)
    print('quicksort:',num_quicksort)
