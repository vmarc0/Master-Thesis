def coloring(near,c, node):
    n = near[node][:]
    flag1 = 0
    flag2 = 0
    for elem in n:
        if (c[elem] == 1):
            c[node] = 2
            flag1 = 1
        if (c[elem] == 2):
            c[node] = 1
            flag2 = 1
    if not (flag1 or flag2):
        c[node] = 1
        flag1 = 1
    if (flag1 and flag2):
        return False
    else:
        return True

# written only for Q = 2
def is_col (gr,d,near,m,n): 
    result = True
    col = torch.zeros(n)
    k = 0
    queue = [k]

    
    for i in range(n):
        if (d[i] == 0):
            col[i] = 1
            
            
    while (len(queue) > 0 and result == True):
        x = queue[-1]
        queue.pop()
        to_col = near[x][:]
        result = coloring(near,col,x)
        for node in to_col:
            if (col[node] == 0 and (node not in queue)):
                queue.append(node)
        if len(queue) == 0:
            for i in range(n):
                if (col[i] == 0 and (i not in queue)):
                    queue.append(i)
                    break
    
    if result:
        for i in range(m):      #check of the links
            node1 = gr[2*i]
            node2 = gr[2*i+1]
            if (col[node1] == col[node2]):
                result = False

    return result
