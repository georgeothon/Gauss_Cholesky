#Bibliotecas
import numpy as np
import random
import time 

def Hilbert_matrix(n):
    
    #Matrix de Hilbert
    A = [[ 1/(i+j-1) for i in range(1,n+1)] for j in range(1,n+1) ]
    
    return np.array(A)

def SemPivoteamento(A,b):
    
    #Matrix nxn
    n = len(A)
    
    #Determinante
    det = 1
    
    #Adiciona o vetor b no final da matriz A e salva em mat 
    mat = np.vstack((A.T,b)).T
    
    for i in range(1,n+1):
        
        # Encontra o menor p tal que i <= p <= n e mat[p,i] != 0
        p = i
        while p <= n :
            if mat[p-1,i-1] != 0:
                break
                
            else:
                p += 1
                
        #Verifica se existe algum zero na linha
        if p > n :
            print('no unique solution exists')
            return
        
        #Troca as linhas p e i
        if p != i:
            
            aux = mat[i].copy()
            mat[i] = mat[p]
            mat[p] = aux
            
            det = -det
            
            
        for j in range(i+1,n+1):
            
            #Multiplicador ji
            m = mat[j-1,i-1]/mat[i-1,i-1]
            
            #Opera nas linhas
            mat[j-1,:] = mat[j-1,:] - m*mat[i-1,:] #Usamos i para pegar linha seguinte

    #Verifica a existencia de soluções
    if mat[n-1,n-1] == 0:
        print('no unique solutions exists')
        return
    
    #Soluções do sistema (Backward substituition)
    x_n = mat[n-1,n]/mat[n-1,n-1]

    X = np.zeros(n)
    X[-1] = x_n
    
    #substituição reversa
    for k in range(n-2,-1,-1):
        
        #produto escalar entre a linha k da matriz e o vetor X
        soma_linha = np.dot(mat[k,k+1:-1],X[k+1:])
        
        #Salva o valor de X_k no array X
        X[k] = ( mat[k,n] - soma_linha )/mat[k,k]
        
    #Calcula o determinante
    for k in range(n):
        det *= mat[k,k]
        
    return X,det
            


def PivoteamentoParcial(A,b):
    
    #Matrix nxn
    n = len(A)
    
    #Adiciona o vetor b no final da matriz A e salva em mat 
    mat = np.vstack((A.T,b)).T
    
    #Determinante
    det = 1
    
    for i in range(n):
        
        #Encontra o pivo e seu indice
        coluna = list(abs(mat[i:,i])) 
        pivo = max(coluna) #maximo em modulo
        indice_pivo = coluna.index(pivo) + i
        
        
        if pivo == 0 :
            print('no unique solutions exits')
            return
        
        #Verifica se o pivo está no topo
        if indice_pivo != i:
            #Troca linhas 
            aux = mat[i].copy()
            mat[i] = mat[indice_pivo]
            mat[indice_pivo] = aux
            
            det = -det
        
        for j in range(i+1,n):
            
            m = mat[j,i]/mat[i,i]
            mat[j] = mat[j] - m * mat[i]
    
    
    #Soluções do sistema (Backward substituition)
    x_n = mat[n-1,n]/mat[n-1,n-1]

    X = np.zeros(n)
    X[-1] = x_n
    
    #substituição reversa
    for k in range(n-2,-1,-1):
        
        #produto escalar entre a linha k da matriz e o vetor X
        soma_linha = np.dot(mat[k,k+1:-1],X[k+1:])
        
        #Salva o valor de X_k no array X
        X[k] = ( mat[k,n] - soma_linha )/mat[k,k]
        
        
    #Calcula o determinante
    for k in range(n):
        det *= mat[k,k]
        
    return X,det

def Cholesky(A,b):
    
    #Matrix nxn
    n = len(A)
    
    #Matrix nxn de zeros
    L = np.zeros((n,n))
    
    det = 1

    
    for i in range(n):
        for j in range(i+1):
            
            #somatório
            soma = sum(L[i,k] * L[j,k] for k in range(j))
            
            #Diagonal principal
            if i == j:
                L[i,j] = np.sqrt(A[i,i] - soma)
                
                #Calcula o determinante
                det *= L[i,j] 
                
            #Abaixo da diagonal principal    
            else:
                L[i,j] = ( A[i,j] - soma )/ L[j,j]
    
    # det(L * L^t) = det(L) * det(L^t) = det(L)^2
    det = det**2
             
    
    
    #Vetores de solução X e Y
    X = np.zeros(n)
    Y = np.zeros(n)
    
    
    #Substiituição direta
    #L * y = b
    
    Y[0] = b[0]/L[0,0]
    
    for k in range(1,n):
        Y[k] = (b[k] - np.dot(L[k,:k],Y[:k]))/L[k,k]
    
    #L Transposta
    Lt = L.T
    
    #Substituição reversa
    # L^t * x = y
    
    X[-1] = Y[-1]/Lt[n-1,n-1]
    
    for k in range(n-2,-1,-1):
        X[k] = (Y[k] - np.dot(Lt[k,k+1:],X[k+1:]))/Lt[k,k]
        
    
    return X,det

def erro(X_predict):
    
    n = len(X_predict)
    X = np.ones(n)
    
    erro = (X - X_predict)**2
    
    return np.sqrt(erro.sum())

def Random_matrix_generator(n):
    
    #Gera matrix com valores escolhidos aleatóriamente no interval [-10,10]
    M = np.array([[random.randint(-10,10) for i in range(n) ] for j in range(n) ])
    #Multiplica pela sua transporta para obter uma matrix A positiva definida
    A = np.dot(M,M.T)

    return A

def main():
    
    
    print('==================================================')    
    print('\n ==================== PARTE I ====================\n')
    print('==================================================')
    
    #Rodando a Parte I
    for k in range(1,5):
        
        #Tamanho do sistema
        n = 2**k
        
        #Sistema à ser resolvido
        A = Hilbert_matrix(n)
        b = [ sum(a) for a in A ]
        
        print('\n================================================\n') 
        print(f'Matriz de Hilbert {n}x{n} \n\n')
        print(np.matrix(A))
        
        #Sem pivotemaneto
        print('\n================================================\n')        
        print('Eliminação de Gauss sem pivoteamento\n')
        print('================================================\n')
        
        X_sem_pivo,det_sem_pivo = SemPivoteamento(A,b)
        
        print(f'Solução: {X_sem_pivo}\n')
        print(f'Erro: {erro(X_sem_pivo)}\n')
        print(f'Determinante: {det_sem_pivo}' )
        
        #Pivoteamento parcial
        print('\n================================================\n')        
        print('Eliminação de Gauss com pivoteamento parcial\n')
        print('================================================\n')
        
        X_com_pivo,det_com_pivo = PivoteamentoParcial(A,b)
        
        print(f'Solução: {X_com_pivo}\n')
        print(f'Erro: {erro(X_com_pivo)}\n')
        print(f'Determinante: {det_com_pivo}' )
        


            
        
    print('\n==================================================')    
    print('\n==================== PARTE II ====================\n')
    print('==================================================')
    
    #Rodando Parte II
    for k in range(1,5):
        
        while True:
            
            n = 2**k
            
            A = Random_matrix_generator(n)
            b = [ sum(a) for a in A ]
        
        
            try:
                
                t0 = time.time()
    
                X_cholesky,det_cholesky = Cholesky(A,b)
    
                t1 = time.time()
    
                X_sem_pivo_2,det_sem_pivo_2 = SemPivoteamento(A,b)
    
                t2 = time.time()
    
                X_numpy = np.linalg.solve(A,b)
    
                t3 = time.time()
                
                det_numpy = np.linalg.det(A)
                
                
                break
                
        print(f'\n================================================\n') 
        print(f'Matriz de A {n}x{n} \n\n')
        print(np.matrix(A))
        
        
        print('\n================================================\n')        
        print('Fatorização de Cholesky\n')
        print('================================================\n')
        
        print(f'Solução: {X_cholesky}\n')
        print(f'Erro: {erro(X_cholesky)}\n')
        print(f'Determinante: {det_cholesky}' )

        print('\n================================================\n')        
        print('Eliminação de Gauss sem pivoteamento\n')
        print('================================================\n')
        
        print(f'Solução: {X_sem_pivo_2}\n')
        print(f'Erro: {erro(X_sem_pivo_2)}\n')
        print(f'Determinante: {det_sem_pivo_2}' )
        
        print('\n================================================\n')        
        print('Comando linalg.solve\n')
        print('================================================\n')

        print(f'Solução: {X_numpy}\n')
        print(f'Erro: {erro(X_numpy)}\n')
        print(f'Determinante: {det_numpy}' )
        
main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    