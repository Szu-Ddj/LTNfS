# '''
# ( 1
# ) 2
# ¬ 3
# ∨ 4
# ∧ 5
# → 6
# '''

# '''
# ∀ 1
# ∃ 2
# ( 1
# ) 2
# ¬ 3
# ∨ 4
# ∧ 5
# → 6
# '''
# import torch
# def evaluate_expression(expression,states,andnet,ornet,notnet):
# # def evaluate_expression(expression,states):
#     def process_expression(expr,sts):
#         stack = torch.Tensor().cuda()
#         stacksts = torch.Tensor().cuda()
#         i = 0
#         result = states
#         while i < len(expr):
#             # print(expr)
#             # print(expr[i])
#             if expr[i] == 1:
#                 stack = torch.concat((stack,torch.tensor([1]).cuda()))
#             elif expr[i] == 2:
#                 sub_expr = torch.Tensor().cuda()
#                 sub_sts = torch.Tensor().cuda()
#                 # print(stack,expr,len(sts),len(expr))
#                 while stack[-1].item() != 1:
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
#         result = torch.mean(result,keepdim=True,dim=0)
#         return result

#     def evaluate_sub_expression(expr,sts):
#         opera_list = [4,5,6]
#         prei = -1
#         result = sts
#         nosigns = False
#         for _index in range(expr.size(0)):
#             _eitem = expr[_index]
#             _item = sts[_index]
#             if _eitem == 3:
#                 nosigns = True
#             if _eitem in opera_list:
#                 if nosigns:
#                     if expr[prei+1:_index][0] != 3:
#                         print(expr)
#                         print(expr[prei+1:_index])
#                     # value = 1 - torch.mean(sts[prei+2:_index],0,keepdim=True)
#                     value = notnet(torch.mean(sts[prei+2:_index],0,keepdim=True))
#                     # value = torch.nn.LeakyReLU(-)
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

#         return result,torch.ones(result.size(0)).cuda() * -1
#     def operate(left,right,signs):
#         left = torch.sigmoid(left)
#         right = torch.sigmoid(right)

#         if signs == 5:
#             # print('and',andnet(left,right).size())
#             return andnet(left,right)
#             # return torch.mul(left,right)
#         if signs == 4:
#             # print('or',ornet(left,right))
#             return ornet(left,right)
#             # return left + right - torch.mul(left,right)
#         if signs == 6:
#             # print('implie',ornet(notnet(left),right))
#             return ornet(notnet(left),right)
#             # return 1 - left + torch.mul(left,right)

#     return process_expression(expression,states)
'''
( 1
) 2
¬ 3
∨ 4
∧ 5
→ 6
'''

'''
∀ 1
∃ 2
( 1
) 2
¬ 3
∨ 4
∧ 5
→ 6
'''
import torch
def evaluate_expression(expression,states,andnet,ornet,notnet):
    def process_expression(expr,sts):
        stack = torch.Tensor().cuda()
        stacksts = torch.Tensor().cuda()
        i = 0
        result = states
        while i < len(expr):
            # print(expr)
            # print(expr[i])
            if expr[i] == 1:
                stack = torch.concat((stack,torch.tensor([1]).cuda()))
            elif expr[i] == 2:
                sub_expr = torch.Tensor().cuda()
                sub_sts = torch.Tensor().cuda()
                # print(stack,expr,len(sts),len(expr))
                while stack[-1].item() != 1:
                    sub_expr = torch.concat((stack[-1:],sub_expr),0)
                    sub_sts = torch.concat((stacksts[-1:],sub_sts),0)
                    stack = stack[:-1]
                    stacksts = stacksts[:-1]
                stack = stack[:-1]
                result,rs = evaluate_sub_expression(sub_expr,sub_sts)
                stack = torch.concat((stack,rs),0)
                stacksts = torch.concat((stacksts,result),0)
            else:
                stack = torch.concat((stack,expr[i:i+1]),0)
                stacksts = torch.concat((stacksts,sts[i:i+1]),0)
            i += 1
        # print('xx',result.size())
        result = torch.mean(result,keepdim=True,dim=0)
        return result

    def evaluate_sub_expression(expr,sts):
        opera_list = [4,5,6]
        prei = -1
        result = sts
        nosigns = False
        for _index in range(expr.size(0)):
            _eitem = expr[_index]
            _item = sts[_index]
            if _eitem == 3:
                nosigns = True
            if _eitem in opera_list:
                if nosigns:
                    if expr[prei+1:_index][0] != 3:
                        print(expr)
                        print(expr[prei+1:_index])
                    value = 1 - torch.mean(sts[prei+2:_index],0,keepdim=True)
                    nosigns = False
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

        return result,torch.ones(result.size(0)).cuda() * -1
    def operate(left,right,signs):
        left = torch.sigmoid(left)
        right = torch.sigmoid(right)
            
            
        if signs == 5:
            return torch.mul(left,right)
        if signs == 4:
            return left + right - torch.mul(left,right)
        if signs == 6:
            return 1 - left + torch.mul(left,right)
        
    def operate2(left,right,signs):

# ∨ 4
# ∧ 5
# → 6
            
        if signs == 5:
            # ab
            return torch.mul(left,right)
        if signs == 4:
            # a + b - ab
            return left + right - torch.mul(left,right)
        if signs == 6:
            # 1 - a + ab
            return 1 - left + torch.mul(left,right)

        

    return process_expression(expression,states)