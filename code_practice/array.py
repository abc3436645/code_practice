from typing import List


# 两个有序数组求交集
def intersect(nums1: List[int], nums2: List[int]):
    """思路：1、设定两个为0的指针，比较两个指针的元素是否相等。如果指针的元素相等，我们将两个指针一起
            向后移动，并且将相等的元素放入空白数组。下图中我们的指针分别指向第一个元素，判断元素相等之
            后，将相同元素放到空白的数组
            2、如果两个指针的元素不相等，我们将小的一个指针后移。图中我们指针移到下一个元素，判断不相
            等之后，将元素小的指针向后移动，继续进行判断。
            3、重复以上步骤

    Args:
        nums1 (List[int]): [1,2,2,3,4]
        nums2 (List[int]): [1,2,3,4]

    Returns:
        _type_: _description_
    """
    left, right = 0, 0
    result = []
    if not nums1 or not nums2:
        return result
    while left < len(nums1) and right < len(nums2):
        if nums1[left] == nums2[right]:
            result.append(nums1[left])
            left += 1
            right += 1
        elif nums1[left] < nums2[right]:
            left += 1
        elif nums1[left] > nums2[right]:
            right += 1

    return result


# 最长公共前缀
"""
输入: ["flower","flow","flight"]
输出: "fl"

输入: ["dog","racecar","car"]
输出: ""
"""


def longest_common_prefix(strs: List[str]):
    """思路：如果strings.Index(x1,x) == 0，则直接跳过（因为此时x就是x1的最长公共前缀），对比下一个元
    素。（如flower和flow进行比较）
    如果strings.Index(x1,x) != 0, 则截取掉基准元素x的最后一个元素，再次和x1进行比较，直至满足
    string.Index(x1,x) == 0，此时截取后的x为x和x1的最长公共前缀。（如flight和flow进行比较，依
    次截取出flow-flo-fl，直到fl被截取出，此时fl为flight和flow的最长公共前缀）

    Args:
        strs (List[str]): _description_

    Returns:
        _type_: _description_
    """
    if len(strs) < 1:
        return ""
    prefix = strs[0]
    for k in strs:
        while k.find(prefix) != 0:
            if len(prefix) == 0:
                return ""
            prefix = prefix[: len(prefix) - 1]
    return prefix


# leetcode189 旋转数组
"""
给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]

输入: [-1,-100,3,99] 和 k = 2
输出: [3,99,-1,-100]

"""


def rotate(nums: List[int], k: int):
    """思路：先对整个nums进行反转，其次按照k划分成2个数组，再分别反转

    Args:
        nums (List[int]): _description_
        k (int): _description_

    Returns:
        _type_: _description_
    """
    reversed_nums = reverse(nums)
    reverse_k = reverse(reversed_nums[:k])
    reversed_left = reverse(reversed_nums[k:])

    return reverse_k + reversed_left


def reverse(arr: List[int]):
    for i in range(len(arr) // 2):
        arr[i], arr[len(arr) - i - 1] = arr[len(arr) - i - 1], arr[i]

    return arr


if __name__ == "__main__":
    nums1 = [1, 2, 2, 4]
    nums2 = [1, 2, 3, 4]

    print(intersect(nums1, nums2))

    print(longest_common_prefix(["flow", "flower", "flop"]))
    print(rotate(nums=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], k=3))
