#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define max_function(a,b) ((a)>(b)?(a):(b))
#define M 64
#define N 128
#define max_iter 20000

double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
float u_acc(float x,float y){
    return (1.0 - pow(x,2))*(1.0 - pow(y,2));
}
float f(float x,float y){
    
    return 2*(1.0 - pow(x,2)) + 2*(1.0 - pow(y,2));
}
int main(int argc, char *argv[]){
    
    float dx, dy;
    
    printf("run Jacobi with gird size: M = %d N = %d max_iter = %d ...\n", M, N, max_iter);

    double time0, time1;
    
    float bound[2][2] = {{-1.0,1.0},{-1.0,1.0}};
    int i,j;
    float x,y;
    
    
    
    float u_new[M*N],u_old[M*N],f_1d[M*N],u[M*N];
    float A[M*N][M*N];
    dx = ((bound[0][1] - bound[0][0])/ (N - 1));//重点注意x方向的剖分是N
    dy = ((bound[1][1] - bound[1][0])/ (M - 1)); //y方向的剖分才是M
    float r1 = -0.5,r2 = -pow(dx/dy,2),r3 = -pow(dy/dx,2),r = 2*(1 - r2 - r3);
    time0 = get_walltime();

    
    for (i = 0; i < M; i++) {
	    for (j = 0; j < N; j++) {
	        x = bound[0][0] + i*dx,y = bound[1][0] + j*dy;
            u_new[i*N + j] = 0.0;
            u[i*N + j] = u_acc(x,y);
	        f_1d[i*N + j] = 0.0;
            A[i*N + j][i*N + j] = 0;
	    }
    }
    for(i = 0; i < M; i++){
        for(j = 0; j < N; j++){
            x = bound[0][0] + i*dx,y = bound[1][0] + j*dy;
            if (i == 0 || i == M - 1 || j == 0 || j == N - 1){
                u_new[i*N + j] = u_acc(x,y);
                A[i*N + j][i*N + j] = 1.0;
            }
            else {
                f_1d[i*N + j] = f(x,y)*(dx*dx + dy*dy);
                A[i*N + j][(i - 1)*N + j - 1] = r1;
                A[i*N + j][(i - 1)*N + j] = r3;
                A[i*N + j][(i - 1)*N + j + 1] = r1;
                A[i*N + j][i*N + j - 1] = r2;
                A[i*N + j][i*N + j] = r;
                A[i*N + j][i*N + j + 1] = r2;
                A[i*N + j][(i + 1)*N + j - 1] = r1;
                A[i*N + j][(i + 1)*N + j] = r3;
                A[i*N + j][(i + 1)*N + j + 1] = r1;
            }
        }
    }

    
//solve
    
    int k = 0;
    
    float error = 0,L1_err = 0,eps = 1e-10; 
    
    float resid;
    int nthreads;
    while (k < max_iter) {
	    error = 0;
        L1_err = 0;
	    #pragma omp parallel for default(shared) private(i,j)
	        //该并行区只有一个for循环，使用#pragma omp parallel for可以自动分配线程处理任务
       	    for(i = 0; i < M; i++){
	            for (j = 0;j < N;j++){
		            u_old[i*N + j] = u_new[i*N + j];
		        }
	        }
        #pragma omp parallel default(shared) private(i,j,resid) \
        reduction(+:error) reduction(max:L1_err)
	    {
	        nthreads = omp_get_num_threads();//获取当前使用线程数目
            #pragma omp for
            for(i = 0; i < M; i++){
		        for (j = 0; j < N; j++){
                    if (i == 0 || i == M - 1 || j == 0 || j == N - 1){
                        continue;
                    }
                    else {
                        resid = f_1d[i*N + j] - (A[i*N + j][(i - 1)*N + j - 1]*u_old[(i - 1)*N + j - 1] + \
                        A[i*N + j][(i - 1)*N + j ]*u_old[(i - 1)*N + j] + \
                        A[i*N + j][(i - 1)*N + j + 1]*u_old[(i - 1)*N + j + 1] + \
                        A[i*N + j][i*N + j - 1]*u_old[i*N + j - 1] + \
                        A[i*N + j][i*N + j]*u_old[i*N + j] + \
                        A[i*N + j][i*N + j + 1]*u_old[i*N + j + 1] + \
                        A[i*N + j][(i + 1)*N + j - 1]*u_old[(i + 1)*N + j - 1] + \
                        A[i*N + j][(i + 1)*N + j]*u_old[(i + 1)*N + j] + \
                        A[i*N + j][(i + 1)*N + j + 1]*u_old[(i + 1)*N + j + 1]);
                        u_new[i*N + j] = u_old[i*N + j] + resid/A[i*N + j][i*N + j];
		                error += resid*resid;
                    } 
                    L1_err = max_function(L1_err, fabs(u_new[i*N + j] - u[i*N + j]));  
		        }
	        }
        }
        
	    error  = sqrt(error)/(M*N);
        if (error < eps)
        {
            break;
        }
        
        
        if (k%1000 == 0){
                
	        printf("%d iteration,error: %3e,L1_err:%.3e\n",k,error,L1_err);
            
	    }
        
        
	    k = k + 1;
        
    }
    time1 = get_walltime() - time0;
    nthreads = omp_get_max_threads();
    printf("Finish at %d,use time:%.2f,use threads:%d, the resid:%.2e,the err:%.2e\n",k,time1,nthreads,error,L1_err);
    
    return 0;
}





