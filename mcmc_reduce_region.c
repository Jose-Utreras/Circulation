#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include <string.h>
#include <omp.h>

#define PI 3.14159265
#define tiny 1e-16

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

double** two_column_file(char *filename, int Nrows){
  int i;
  int j;
  double** mat=malloc(Nrows*sizeof(double*));
  for(i=0;i<Nrows;++i)
  mat[i]=malloc(2*sizeof(double));
  FILE *file;
  file=fopen(filename, "r");

  for(i = 0; i < Nrows; i++){
    for(j = 0; j < 2; j++) {
       if (!fscanf(file, "%lf", &mat[i][j]))
          break;}}
  fclose(file);
  return mat;}

char* concat(const char *s1, const char *s2)
  {
      char *result = malloc(strlen(s1)+strlen(s2)+1);//+1 for the null-terminator
      //in real code you would check for errors in malloc here
      strcpy(result, s1);
      strcat(result, s2);
      return result;
  }

double sin_delta(double p, double q,double delta){
  double result;
  result=sin(PI*p*delta)*sin(PI*q*delta)/(delta*delta*PI*PI);
  result=result/((p+tiny)*(q+tiny));
  result=result*result;
  return result;
}

double gama(double p, double n1, double n2, double pc, double pmin, double pmax){
  if (p<pmin) return 0;
  if (p>pmax) return 0;
  if (p<pc) return 1.0/pow(p,n1);
  else return pow(pc,n2-n1)/pow(p,n2);
}

double func1(double r, double n1, double n2, double pmin, double pc ,double f1){
  double result;
  if (r<f1){
    result=pow(pow(pmin,2.0-2.0*n1)+(2.0-2.0*n1)*r,1.0/(2.0-2.0*n1));}
  else{
    result=pow(pc,2.0-2.0*n2)+(2.0-2.0*n2)*(r-f1)*pow(pc,2.0*(n1-n2));
    result=pow(result,1.0/(2.0-2.0*n2));}
  return result;
}

double func2(double r, double n1, double n2, double pmin, double pc ,double f1){
  double result;
  if (r<f1){
    result=pmin*exp(r);}
  else{
  result=pow(pc,2.0-2.0*n2)+(2.0-2.0*n2)*(r-f1)*pow(pc,2.0*(n1-n2));
  result=pow(result,1.0/(2.0-2.0*n2));}
  return result;
}

double func3(double r, double n1, double n2, double pmin, double pc ,double f1){
  double result;
  if (r<f1){
    result=pow(pow(pmin,2.0-2.0*n1)+(2.0-2.0*n1)*r,1.0/(2.0-2.0*n1));}
  else{
    result=pc*exp((r-f1)*pow(pc,2*(n1-n2)));}
  return result;}

double func4(double r, double n1, double n2, double pmin, double pc ,double f1){
  double result;
  if (r<f1){
    result=pmin*exp(r);}
  else{
    result=pc*exp((r-f1)*pow(pc,2*(n1-n2)));}
  return result;}

float *sigmas(double n1, double n2, double pc, double pmin, double pmax, int Nbins, float *delta){
  n1=n1-1;
  n2=n2-1;
  //srand(time(0));
  int Npoints;
  float random,angle;
  double f1,f2,g1,g2,A,B,p,q,sum;
  double maximo=1.0*RAND_MAX;
  float *result = malloc(sizeof(float) * Nbins);

  if (n1!=1.0 && n2!=1.0){
    f1=pow(pc,2.0-2.0*n1)/(2.0-2.0*n1) - pow(pmin,2.0-2.0*n1)/(2.0-2.0*n1);
    f2=pow(pmax,2.0-2.0*n2)-pow(pc,2.0-2.0*n2);
    f2=f2*pow(pc,2.0*(n2-n1))/(2.0-2.0*n2);

    //These are the condition of the index for the velocity field
    if (n1!=0.0 && n2!=0.0){
      g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);
      g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
      g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);}
    else if(n1!=0.0){
      g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);
      g2=log(pmax/pc)*pow(pc,2.0*(n2-n1));}
    else if(n2!=0){
      g1=log(pc/pmin);
      g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
      g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);}
    else {
      g1=log(pc/pmin);
      g2=log(pmax/pc)*pow(pc,2.0*(n2-n1));}

    A=f1+f2;
    B=g1+g2;


    for(int j = 0; j < Nbins; j++){
      sum=0.0;
      Npoints= 10+(int) 800*(pmax-pmin)*delta[j];
      for(int i = 0; i < Npoints; i++){
        random = rand()/maximo;
        angle  = 0.5*PI*rand()/maximo;
        random = func1(random*A,n1,n2,pmin,pc,f1);
        p=random*cos(angle);
        q=random*sin(angle);
        sum=sum+sin_delta(p,q,delta[j]);}
      result[j]=2*PI*sqrt(A*sum/(1.0*B*Npoints));}
  }
  else if (n1==1.0 && n2!=1.0){
    f1=log(pc/pmin);
    f2=pow(pmax,2.0-2.0*n2)-pow(pc,2.0-2.0*n2);
    f2=f2*pow(pc,2.0*(n2-n1))/(2.0-2.0*n2);
    g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);

    if (n2!=0.0){
      g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
      g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);}
    else {g2=log(pmax/pc)*pow(pc,2.0*(n2-n1));}

    A=f1+f2;
    B=g1+g2;

    for(int j = 0; j < Nbins; j++){
      sum=0.0;
      Npoints= 10+(int) 800*(pmax-pmin)*delta[j];
      for(int i = 0; i < Npoints; i++){
        random = rand()/maximo;
        angle  = 0.5*PI*rand()/maximo;
        random = func1(random*A,n1,n2,pmin,pc,f1);
        p=random*cos(angle);
        q=random*sin(angle);
        sum=sum+sin_delta(p,q,delta[j]);}
      result[j]=2*PI*sqrt(A*sum/(1.0*B*Npoints));}

  }
  else if (n1!=1.0 && n2==1.0){
    f1=pow(pc,2.0-2.0*n1)/(2.0-2.0*n1) - pow(pmin,2.0-2.0*n1)/(2.0-2.0*n1);
    f2=log(pmax/pc)*pow(pc,2.0*(n2-n1));
    g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
    g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);

    if (n1!=0.0){g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);}
    else {g1=log(pc/pmin);}

    A=f1+f2;
    B=g1+g2;

    for(int j = 0; j < Nbins; j++){
      sum=0.0;
      Npoints= 10+(int) 800*(pmax-pmin)*delta[j];
      for(int i = 0; i < Npoints; i++){
        random = rand()/maximo;
        angle  = 0.5*PI*rand()/maximo;
        random = func1(random*A,n1,n2,pmin,pc,f1);
        p=random*cos(angle);
        q=random*sin(angle);
        sum=sum+sin_delta(p,q,delta[j]);}
      result[j]=2*PI*sqrt(A*sum/(1.0*B*Npoints));}
  }
  else if (n1==1.0 && n2==1.0){
    f1=log(pc/pmin);
    f2=log(pmax/pc)*pow(pc,2.0*(n2-n1));
    g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);
    g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
    g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);

    A=f1+f2;
    B=g1+g2;

    for(int j = 0; j < Nbins; j++){
      sum=0.0;
      Npoints= 10+(int) 800*(pmax-pmin)*delta[j];
      for(int i = 0; i < Npoints; i++){
        random = rand()/maximo;
        angle  = 0.5*PI*rand()/maximo;
        random = func1(random*A,n1,n2,pmin,pc,f1);
        p=random*cos(angle);
        q=random*sin(angle);
        sum=sum+sin_delta(p,q,delta[j]);}
      result[j]=2*PI*sqrt(A*sum/(1.0*B*Npoints));}}
  return result;
  }

double one_sigma(double n1, double n2, double pc, double pmin, double pmax){
    n1=n1-1;
    n2=n2-1;
    //srand(time(0));
    double f1,f2,g1,g2,A,B,p,q,sum;
    double result;

    if (n1!=1.0 && n2!=1.0){
      f1=pow(pc,2.0-2.0*n1)/(2.0-2.0*n1) - pow(pmin,2.0-2.0*n1)/(2.0-2.0*n1);
      f2=pow(pmax,2.0-2.0*n2)-pow(pc,2.0-2.0*n2);
      f2=f2*pow(pc,2.0*(n2-n1))/(2.0-2.0*n2);

      //These are the condition of the index for the velocity field
      if (n1!=0.0 && n2!=0.0){
        g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);
        g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
        g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);}
      else if(n1!=0.0){
        g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);
        g2=log(pmax/pc)*pow(pc,2.0*(n2-n1));}
      else if(n2!=0){
        g1=log(pc/pmin);
        g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
        g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);}
      else {
        g1=log(pc/pmin);
        g2=log(pmax/pc)*pow(pc,2.0*(n2-n1));}

      A=f1+f2;
      B=g1+g2;

      result=2*PI*sqrt(A/(1.0*B));
    }
    else if (n1==1.0 && n2!=1.0){
      f1=log(pc/pmin);
      f2=pow(pmax,2.0-2.0*n2)-pow(pc,2.0-2.0*n2);
      f2=f2*pow(pc,2.0*(n2-n1))/(2.0-2.0*n2);
      g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);

      if (n2!=0.0){
        g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
        g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);}
      else {g2=log(pmax/pc)*pow(pc,2.0*(n2-n1));}

      A=f1+f2;
      B=g1+g2;

      result=2*PI*sqrt(A/(1.0*B));

    }
    else if (n1!=1.0 && n2==1.0){
      f1=pow(pc,2.0-2.0*n1)/(2.0-2.0*n1) - pow(pmin,2.0-2.0*n1)/(2.0-2.0*n1);
      f2=log(pmax/pc)*pow(pc,2.0*(n2-n1));
      g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
      g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);

      if (n1!=0.0){g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);}
      else {g1=log(pc/pmin);}

      A=f1+f2;
      B=g1+g2;

      result=2*PI*sqrt(A/(1.0*B));
    }
    else if (n1==1.0 && n2==1.0){
      f1=log(pc/pmin);
      f2=log(pmax/pc)*pow(pc,2.0*(n2-n1));
      g1=pow(pc,-2.0*n1)/(-2.0*n1) - pow(pmin,-2.0*n1)/(-2.0*n1);
      g2=pow(pmax,-2.0*n2)-pow(pc,-2.0*n2);
      g2=g2*pow(pc,2.0*(n2-n1))/(-2.0*n2);

      A=f1+f2;
      B=g1+g2;

      result=2*PI*sqrt(A/(1.0*B));}
    return result;
    }

double uniform(double minimo, double maximo){
  //srand(time(NULL));
  return minimo+(maximo-minimo)*rand()/(1.0*RAND_MAX);
}

double probabilities(double n1 , double n2, double pc,
  double pmin, double pmax, int Nbins, double dx,float *sigma, float *delta, float *error){
  //srand(time(NULL));
  double fc,prob;
  float *temp;


  temp=sigmas(n1, n2, pc, pmin, pmax, Nbins, delta); // still need to add dx factor
	//fc=uniform(0.9*temp[0]*dx*dx/sigma[0],1*temp[0]*dx*dx/sigma[0]);
	fc=temp[0]*dx*dx/sigma[0];
	prob=0.0;

	for(int i=0;i<Nbins;i++){
		prob=prob-0.5*(pow((sigma[i]-fc*temp[i]*dx*dx)/error[i],2)+2*log(error[i]));
		}
	free(temp);
	return prob;

}


int main(int argc, char* argv[]) {
srand(time(0));

/* the arguments of the function I need are the  following:
n1,n2,pc,pmin,pmax,delta
*/
char *name;
int N, Nres, Nbins;
double L;


sscanf(argv[1],"%lf",&L);
sscanf(argv[2],"%d",&N);
Nbins=(argc-3)/3;
double pmin,pmax,dx;

pmin=4.0/L;
pmax=(1.0*N-2.0)/(4.0*L);
dx=L/(1.0*N);

float sigma[Nbins];
float delta[Nbins];
float error[Nbins];

for(int i = 0; i < Nbins; i++){
 sscanf(argv[3+i],"%f",&sigma[i]);
 sscanf(argv[3+Nbins+i],"%f",&delta[i]);
 sscanf(argv[3+2*Nbins+i],"%f",&error[i]);}


double n1_min,n1_max,n2_min,n2_max,n1,n2,pc,maximo,aux,pc_min,pc_max;
n1_min=-1.0;
n1_max=3.0;
n2_min=-1.0;
n2_max=3.0;
pc_min=pmin;
pc_max=pmax*0.5;
int Nn = 7.0;
int Np = 25.0;
double probs[Np];
double pc_array[Np];
double n1_array[Np];
double n2_array[Np];
/*
IN THIS PART A REDUCE THE PARAMETER SPACE FOR PC
*/
for(int i=0;i<Np;i++){
	pc=pc_min+1.0*i*(pc_max-pc_min)/(1.0*Np-1.0);
	pc_array[i]=pc;
	maximo=-1e32;
	for(int j=0;j<Nn;j++){
		for(int k=0;k<Nn;k++){
			n1=n1_min+1.0*j*(n1_max-n1_min)/(1.0*Nn-1.0);
			n2=n2_min+1.0*k*(n2_max-n2_min)/(1.0*Nn-1.0);
			aux=probabilities(n1 , n2 , pc , pmin , pmax, Nbins, dx, sigma, delta, error);
			if(aux>maximo){maximo=aux;}
		}
	}
	probs[i]=maximo;
}

maximo=-1e32;
for(int i=0;i<Np;i++){
	if(probs[i]>maximo){maximo=probs[i];}
}

int lw,up;

for(int i=0;i<Np;i++){
	if(probs[i]>(maximo-8)){lw=max(0,i-1); break;}
}

for(int i=0;i<Np;i++){
	if(probs[Np-1-i]>(maximo-8)){up=min(Np-1,Np-i); break;}
}

pc_min=pc_array[lw];
pc_max=pc_array[up];
printf("%f\t %f\n", pc_min,pc_max);

/*
IN THIS PART A REDUCE THE PARAMETER SPACE FOR N1
*/

for(int i=0;i<Np;i++){
	n1=n1_min+1.0*i*(n1_max-n1_min)/(1.0*Np-1.0);
	n1_array[i]=n1;
	maximo=-1e32;
	for(int j=0;j<Nn;j++){
		for(int k=0;k<Nn;k++){
			pc=pc_min+1.0*j*(pc_max-pc_min)/(1.0*Nn-1.0);
			n2=n2_min+1.0*k*(n2_max-n2_min)/(1.0*Nn-1.0);
			aux=probabilities(n1 , n2 , pc , pmin , pmax, Nbins, dx, sigma, delta, error);
			if(aux>maximo){maximo=aux;}
		}
	}
	probs[i]=maximo;
}

maximo=-1e32;
for(int i=0;i<Np;i++){
	if(probs[i]>maximo){maximo=probs[i];}
}

for(int i=0;i<Np;i++){
	if(probs[i]>(maximo-8)){lw=max(0,i-1); break;}
}

for(int i=0;i<Np;i++){
	if(probs[Np-1-i]>(maximo-8)){up=min(Np-1,Np-i); break;}
}

n1_min=n1_array[lw];
n1_max=n1_array[up];
printf("%f\t %f\n", n1_min,n1_max);

/*
IN THIS PART A REDUCE THE PARAMETER SPACE FOR N2
*/

for(int i=0;i<Np;i++){
	n2=n2_min+1.0*i*(n2_max-n2_min)/(1.0*Np-1.0);
	n2_array[i]=n2;
	maximo=-1e32;
	for(int j=0;j<Nn;j++){
		for(int k=0;k<Nn;k++){
			n1=n1_min+1.0*j*(n1_max-n1_min)/(1.0*Nn-1.0);
			pc=pc_min+1.0*k*(pc_max-pc_min)/(1.0*Nn-1.0);
			aux=probabilities(n1 , n2 , pc , pmin , pmax, Nbins, dx, sigma, delta, error);
			if(aux>maximo){maximo=aux;}
		}
	}
	probs[i]=maximo;
}

maximo=-1e32;
for(int i=0;i<Np;i++){
	if(probs[i]>maximo){maximo=probs[i];}
}

for(int i=0;i<Np;i++){
	if(probs[i]>(maximo-8)){lw=max(0,i-1); break;}
}

for(int i=0;i<Np;i++){
	if(probs[Np-1-i]>(maximo-8)){up=min(Np-1,Np-i); break;}
}

n2_min=n1_array[lw];
n2_max=n2_array[up];
printf("%f\t %f\n", n2_min, n2_max);

}
