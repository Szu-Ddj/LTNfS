# import torch
# '''
# ∀ 47444, 7471
#  x 3023
#  ¬ 5505, 11582
#  ) 4839
#  ( 36
# ( 1640
#  ∨ 47444, 11423 
#  ∧ 47444, 6248 
#  → 42484 
# ∧ ∨ ¬ →
#  '''
# def evaluate_expression(expression,states,andnet,ornet,notnet):
#     def process_expression(expr,sts):
#         stack = torch.Tensor().cuda()
#         stacksts = torch.Tensor().cuda()
#         i = 0
#         result = states
#         while i < len(expr):
#             if expr[i] == 36 or expr[i] == 1640: #1
#                 stack = torch.concat((stack,torch.tensor([36]).cuda())) #1-1
#             elif expr[i] == 4839: #2
#                 sub_expr = torch.Tensor().cuda()
#                 sub_sts = torch.Tensor().cuda()
#                 # print(stack,expr,len(sts),len(expr))
#                 while stack[-1].item() != 36 and stack[-1].item != 1640: #3
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
# #  ∨ 47444, 11423 
# #  ∧ 47444, 6248 
# #  → 42484 
#     def evaluate_sub_expression(expr,sts):
#         opera_list = [11423,6248,42484]  #4
#         prei = -1
#         result = sts
#         nosigns = False
#         for _index in range(expr.size(0)):
#             _eitem = expr[_index]
#             _item = sts[_index]
#             if _eitem == 11582:  #5
#                 if expr[_index-1] != 5505:
#                     print(expr)
#                     assert False
#                 nosigns = True
#             elif _eitem in opera_list:
#                 if (_eitem == 11423 or _eitem == 6248) and expr[_index-1] != 47444:
#                     print(expr)
#                     assert False
#                 if nosigns:
#                     if expr[prei+1:_index][1] != 11582:  #6
#                         print(expr)
#                         print(expr[prei+1:_index])
#                     # value = 1 - torch.mean(sts[prei+3:_index],0,keepdim=True)
#                     value = notnet(torch.mean(sts[prei+3:_index],0,keepdim=True))
#                     nosigns = False
#                 else:
#                     if _eitem == 11423 or _eitem == 6248:
#                         value = torch.mean(sts[prei+1:_index- 1],0,keepdim=True)
#                     else:
#                         value = torch.mean(sts[prei+1:_index],0,keepdim=True)

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
# #  ∨ 47444, 11423 
# #  ∧ 47444, 6248 
# #  → 42484 
#     def operate(left,right,signs):
#         left = torch.sigmoid(left)
#         right = torch.sigmoid(right)
#         if signs == 6248:  #7∧
#             return andnet(left,right)
#             # return torch.mul(left,right)
#         if signs == 11423:  #8∨
#             return ornet(left,right)
#             # return left + right - torch.mul(left,right)
#         if signs == 42484:  #9→
#             return ornet(notnet(left),right)
#             # return 1 - left + torch.mul(left,right)

        

#     return process_expression(expression,states)
import torch
'''
∀ 47444, 7471
 x 3023
 ¬ 5505, 11582
 ) 4839
 ( 36
( 1640
 ∨ 47444, 11423 
 ∧ 47444, 6248 
 → 42484 
∧ ∨ ¬ →
 '''
def evaluate_expression(expression,states,andnet,ornet,notnet):
    def process_expression(expr,sts):
        stack = torch.Tensor().cuda()
        stacksts = torch.Tensor().cuda()
        i = 0
        result = states
        while i < len(expr):
            if expr[i] == 36 or expr[i] == 1640: #1
                stack = torch.concat((stack,torch.tensor([36]).cuda())) #1-1
            elif expr[i] == 4839: #2
                sub_expr = torch.Tensor().cuda()
                sub_sts = torch.Tensor().cuda()
                # print(stack,expr,len(sts),len(expr))

                while stack[-1].item() != 36 and stack[-1].item != 1640: #3
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
#  ∨ 47444, 11423 
#  ∧ 47444, 6248 
#  → 42484 
    def evaluate_sub_expression(expr,sts):
        opera_list = [11423,6248,42484]  #4
        prei = -1
        result = sts
        nosigns = False
        for _index in range(expr.size(0)):
            _eitem = expr[_index]
            _item = sts[_index]
            if _eitem == 11582:  #5
                if expr[_index-1] != 5505:
                    print(expr)
                    assert False
                nosigns = True
            elif _eitem in opera_list:
                if (_eitem == 11423 or _eitem == 6248) and expr[_index-1] != 47444:
                    print(expr)
                    assert False
                if nosigns:
                    if expr[prei+1:_index][1] != 11582:  #6
                        print(expr)
                        print(expr[prei+1:_index])
                    value = 1 - torch.mean(sts[prei+3:_index],0,keepdim=True)
                    nosigns = False
                else:
                    if _eitem == 11423 or _eitem == 6248:
                        value = torch.mean(sts[prei+1:_index- 1],0,keepdim=True)
                    else:
                        value = torch.mean(sts[prei+1:_index],0,keepdim=True)

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
#  ∨ 47444, 11423 
#  ∧ 47444, 6248 
#  → 42484 
    def operate(left,right,signs):
        left = torch.sigmoid(left)
        right = torch.sigmoid(right)
        if signs == 6248:  #7∧
            return torch.mul(left,right)
        if signs == 11423:  #8∨
            return left + right - torch.mul(left,right)
        if signs == 42484:  #9→
            return 1 - left + torch.mul(left,right)

        

    return process_expression(expression,states)