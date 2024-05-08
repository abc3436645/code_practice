"""
回溯模板:核心思路for循环+递归

def backstracking(参数):
    if 终止条件：
        存放结果
        return
    for (选择集合中的元素进行遍历):
        处理节点逻辑
        backstracking(路径选择列表，是否需要index)  ## 递归 + 剪枝优化
        回溯，撤销处理结果
"""


##77组合,给定n和k，返回1-n中所有可能的k个数的组合 
class Solution:
    def combination(self,n:int,k:int):
        res = []
        path = []
        def backtracking(n,k,start_index):
            """回溯函数

            Args:
                n (_type_): _description_
                k (_type_): _description_
                start_index (_type_): 控制循环的起始位置
            """
            if len(path) == k:
                res.append(path[:])
                return 
            for i in range(start_index,n+1): # 模拟列表[1,2,3...n]
                path.append(i)
                backtracking(n,k,i+1)
                path.pop()
        backtracking(n,k,1)
        
        return res

###优化
class Solution:
    def combination(self,n:int,k:int):
        res = [] # 外部变量存放结果
        path = [] # 外部变量存放路径
        def backstracking(n,k,start_index):
            """
            剪枝，如果循环开始的位置往后数，不足我们需要的个数，那么就没有必要接着搜索了
            剪枝操作在for循环中 通过索引进行控制
            """
            if len(path) == k:
                res.append(path.copy())
                return
            for i in range(start_index,n-(k-len(path)+1+1)):   ###剪枝
                path.append(i)
                backstracking(n,k,start_index+1)
                path.pop()
        backstracking(n,k,1)

        return res


###216组合问题III  找出所有相加和为n的k个数的组合，组合中只有1-9的正整数，并且不存在重复数字
##  k = 3,n =9 
### [[1,2,6],[1,3,5],[2,3,4]]
class Solution:
    def combination(self,n:int,k:int):
        res = []
        path = []
        def backstracking(target,k,start_index,path_sum):
            if path_sum > target:
                return
            if len(path) == k:
                if path_sum == target:
                    res.append(path[:])
                    return
            for i in range(start_index,10):
                path_sum += i
                path.append(i)
                backstracking(n,k,i+1,path_sum)
                path.pop()
                path_sum -= i
        backstracking(n,k,1,0)
        
        return res

class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        self.res = []
        self.path = []
        self.backtracking(k,n,1)
        return self.res

    def backtracking(self, k, n, start):
        if sum(self.path) > n:
            return
        if sum(self.path) == n and len(self.path) == k:
            self.res.append(self.path[:])
            return
        for i in range(start, 10):
            self.path.append(i)
            self.backtracking(k, n, i + 1)
            self.path.pop()



### 17电话号码的字母组合
class Solution:
    def letters_combination(self,digits:str,index:int):
        pass