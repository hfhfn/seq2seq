
def sub_max_list(nums):
    sub_nums = []
    sub_nums1 = []
    for i in range(len(nums)):
        sum1 = 0
        sum2 = 0
        index = []
        nums1 = nums[i:]
        sub_nums1.extend(sub_nums)
        # print(sub_nums)
        for k, j in enumerate(nums1):
            sum = sum1
            sum1 = sum + j
            if sum > sum2:
                sum2 = sum
            index.append(j)
            if k == 0:
                sub_nums.append((sum1, index))
            if sum1 > sum2:
                sub_nums = []
                sub_nums.append((sum1, index))
            elif sum1 == sum:
                sub_nums.append((sum1, index))
    sub_nums1.extend(sub_nums)
    sub_nums = [sub_nums1[0]]
    # print(sub_nums1)
    for i in sub_nums1:
        if i[0] > sub_nums[0][0]:
            sub_nums = []
            sub_nums.append(i)
        elif i[0] == sub_nums[0][0] and i[1] != sub_nums[0][1]:
            sub_nums.append(i)
    return sub_nums[0][0], [sub_nums[i][1] for i in range(len(sub_nums))][0]
    # return sub_nums

ret = sub_max_list([-1])
print(ret)