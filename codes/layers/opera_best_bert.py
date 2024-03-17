import torch

def evaluate_expression(expression,states):
    def process_expression(expr,sts):
        stack = torch.Tensor().cuda()
        stacksts = torch.Tensor().cuda()
        i = 0
        result = states
        while i < len(expr):
            if expr[i] == 1006:
                stack = torch.concat((stack,torch.tensor([1006]).cuda()))
            elif expr[i] == 1007:
                sub_expr = torch.Tensor().cuda()
                sub_sts = torch.Tensor().cuda()
                # print(stack,expr,len(sts),len(expr))
                while stack[-1].item() != 1006:
                    sub_expr = torch.concat((stack[-1:],sub_expr),0)
                    sub_sts = torch.concat((stacksts[-1:],sub_sts),0)
                    stack = stack[:-1]
                    stacksts = stacksts[:-1]
                    if len(stack)==0:
                        break
                stack = stack[:-1]
                result,rs = evaluate_sub_expression(sub_expr,sub_sts)
                stack = torch.concat((stack,rs),0)
                stacksts = torch.concat((stacksts,result),0)
            else:
                stack = torch.concat((stack,expr[i:i+1]),0)
                stacksts = torch.concat((stacksts,sts[i:i+1]),0)
            i += 1
        # print('xx',result.size())
        # print('xx',)
        # result = evaluate_sub_expression(sub_expr,sub_sts)
        result = torch.mean(result,keepdim=True,dim=0)
        return result

    def evaluate_sub_expression(expr,sts):
        opera_list = [1603,1602,1585]
        prei = -1
        result = sts
        nosigns = False
        for _index in range(expr.size(0)):
            _eitem = expr[_index]
            _item = sts[_index]
            if _eitem == 1078:
                nosigns = True
            if _eitem in opera_list:
                if nosigns:
                    if expr[prei+1:_index][0] != 1078:
                        value = torch.mean(sts[prei+1:_index],0,keepdim=True)
                    #     print(expr)
                    #     print(expr[prei+1:_index])
                    value = 1 - torch.mean(sts[prei+2:_index],0,keepdim=True)
                    nosigns = False
                else:
                    value = torch.mean(sts[prei+1:_index],0,keepdim=True)
# 22380.,  1996.,  2445.,  6998.,  1998., 25247.,  2582.,  1010.,  2057.,
# 2064., 18547.,  1996.,  2206.,  2034.,  1011.,  2344.,  7961.,  4861.,
# 1024.,  1078.,  2003.,  3407.,  3178.,  1585.,  8699.,  4863.,  1024.,
# 1996.,  6251.,  2515.,  2025.,  4671.,  1037.,  3893.,  2030.,  4997.,
# 7729.,  2875.,  3407.,  3178.,  1010.,  2061.,  1996.,  7729.,  2003.,
# 8699.,  1012.
                if prei != -1:

                    value = operate(prev,value,expr[prei])
                prev = value
                prei = _index

            if _index == len(expr) - 1:
                if prei == -1:

                    break
                # print(prev,sts[prei+1:len(expr)],expr[prei])
                result = operate(prev,torch.mean(sts[prei+1:len(expr)],0,keepdim=True),expr[prei])

        return result,torch.ones(result.size(0)).cuda()*-1
    def operate(left,right,signs):
        left = torch.sigmoid(left)
        right = torch.sigmoid(right)
        if signs == 1602:
            return torch.mul(left,right)
        if signs == 1603:
            return left + right - torch.mul(left,right)
        if signs == 1585:
            return 1 - left + torch.mul(left,right)
    return process_expression(expression,states)
# def evaluate_expression(expression,states):
#     def process_expression(expr,sts):
#         stack = torch.Tensor().cuda()
#         stacksts = torch.Tensor().cuda()
#         i = 0
#         result = states
#         while i < len(expr):
#             if expr[i] == 1006:
#                 stack = torch.concat((stack,torch.tensor([1006]).cuda()))
#             elif expr[i] == 1007:
#                 sub_expr = torch.Tensor().cuda()
#                 sub_sts = torch.Tensor().cuda()
#                 # print(stack,expr,len(sts),len(expr))
#                 while stack[-1].item() != 1006:
#                     sub_expr = torch.concat((stack[-1:],sub_expr),0)
#                     sub_sts = torch.concat((stacksts[-1:],sub_sts),0)
#                     stack = stack[:-1]
#                     stacksts = stacksts[:-1]
#                 stack = stack[:-1]
#                 result,rs = evaluate_sub_expression(sub_expr,sub_sts)
#                 stack = torch.concat((stack,rs),0)
#                 stacksts = torch.concat((stacksts,result),0)
#             else:
#                 stack = torch.concat((stack,expr[i:i+1]),0)
#                 stacksts = torch.concat((stacksts,sts[i:i+1]),0)
#             i += 1
#         # print('xx',result.size())
#         # print('xx',)
#         # result = evaluate_sub_expression(sub_expr,sub_sts)
#         result = torch.mean(result,keepdim=True,dim=0)
#         return result

#     def evaluate_sub_expression(expr,sts):
#         opera_list = [1603,1602,1585]
#         prei = -1
#         result = sts
#         nosigns = False
#         for _index in range(expr.size(0)):
#             _eitem = expr[_index]
#             _item = sts[_index]
#             if _eitem == 1078:
#                 nosigns = True
#             if _eitem in opera_list:
#                 if nosigns:
#                     if expr[prei+1:_index][0] != 1078:
#                         print(expr)
#                         print(expr[prei+1:_index])
#                     value = 1 - torch.mean(sts[prei+2:_index],0,keepdim=True)
#                     nosigns = False
#                 else:
#                     value = torch.mean(sts[prei+1:_index],0,keepdim=True)

#                 if prei != -1:

#                     value = operate(prev,value,expr[prei])
#                 prev = value
#                 prei = _index

#             if _index == len(expr) - 1:
#                 if prei == -1:

#                     break
#                 # print(prev,sts[prei+1:len(expr)],expr[prei])
#                 result = operate(prev,torch.mean(sts[prei+1:len(expr)],0,keepdim=True),expr[prei])

#         return result,torch.ones(result.size(0)).cuda()*-1
#     def operate(left,right,signs):
#         # if 1078 in left:
#         #     left = 1 - torch.mean(left[1:,:],0,keepdim=True)
#         # else:
#         # left = torch.mean(left,0,keepdim=True)

#         # if 1078 in right:
#         #     right = 1 - torch.mean(right[1:,:],0,keepdim=True)
#         # else:
#         # right = torch.mean(right,0,keepdim=True)
            
            
#         if signs == 1602:
#             return torch.mul(left,right)
#         if signs == 1603:
#             return left + right - torch.mul(left,right)
#         if signs == 1585:
#             return 1 - left + torch.mul(left,right)

        

#     return process_expression(expression,states)