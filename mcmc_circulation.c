#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include <string.h>

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
      Npoints= 10+(int) 400*(pmax-pmin)*delta[j];
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
      Npoints= 10+(int) 400*(pmax-pmin)*delta[j];
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
      Npoints= 10+(int) 400*(pmax-pmin)*delta[j];
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
      Npoints= 10+(int) 400*(pmax-pmin)*delta[j];
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

double walker(double n1_min, double n1_max, double n2_min, double n2_max, double f_min,
  double f_max, double pmin, double pmax, int Nbins, float *delta, double st, double sc,
  double **circulation, int Nres, int steps, double dx){
  //srand(time(NULL));
  double n1,n2,pc,fc,aux,prob;
  double n1_n,n2_n,pc_n,fc_n,prob_n;  // new parameters
  float *temp;
  double fc_temp,alpha,rnd;
  int counter=0;

  do{
  n1=uniform(n1_min,n1_max);
  n1=uniform(n1_min,n1_max);
  n2=uniform(n2_min,n2_max);
  pc=uniform(pmin,pmax);

  n1=1.5;
	n2=3.0;
	pc=0.0016;
  fc_temp=one_sigma(n1, n2, pc, pmin, pmax);

	//printf("%f\t %f\t %f\t %f\t %f\t %f\n",n1,n2,pc,pmin,pmax,fc_temp*dx*dx);
	printf("%f\n", sc/(fc_temp*dx*dx));
  //fc=uniform(min(sc/(fc_temp*dx*dx*3),f_max),min(5*sc/(fc_temp*dx*dx*3),f_max));
	fc=uniform(0.9*sc/(fc_temp*dx*dx),1.1*sc/(fc_temp*dx*dx));
  temp=sigmas(n1, n2, pc, pmin, pmax, Nbins, delta);
  prob=-1e-32;
	//printf("%f\t %f\t %f\t  %f\t %f\n", dx,delta[0],st,temp[0],dx*dx*temp[0]);
  for(int j=0; j<Nres;j++){

    aux=erf((circulation[j][0]+0.5*st*pow(dx/delta[(int)(circulation[j][1])],1.5))
    /(fc*sqrt(2.0)*dx*dx*temp[(int)(circulation[j][1])]))
    -erf((circulation[j][0]-0.5*st*pow(dx/delta[(int)(circulation[j][1])],1.5))
    /(fc*sqrt(2.0)*dx*dx*temp[(int)(circulation[j][1])]));
    prob=prob+log(0.5*aux);
    if(aux==0.0)break;
  }
  free(temp);}while(!isfinite(prob));

  do{
  n1_n=uniform(n1_min,n1_max);
  n2_n=uniform(n2_min,n2_max);
  pc_n=uniform(pmin,pmax);
  fc_temp=one_sigma(n1_n, n2_n, pc_n, pmin, pmax);
  fc_n=uniform(min(sc/(fc_temp*dx*dx*3),f_max),min(5*sc/(fc_temp*dx*dx*3),f_max));
  temp=sigmas(n1_n, n2_n, pc_n, pmin, pmax, Nbins, delta);
  prob_n=-1e-32;

  for(int j=0; j<Nres;j++){

    aux=erf((circulation[j][0]+0.5*st*pow(dx/delta[(int)(circulation[j][1])],1.5))
    /(fc_n*sqrt(2.0)*dx*dx*temp[(int)(circulation[j][1])]))
    -erf((circulation[j][0]-0.5*st*pow(dx/delta[(int)(circulation[j][1])],1.5))
    /(fc_n*sqrt(2.0)*dx*dx*temp[(int)(circulation[j][1])]));
    prob_n=prob_n+log(0.5*aux);
    if(aux==0.0)break;
  }
  free(temp);
  printf("%f\t %f\t %f\t  %f\t %f\t %f\t %f\n",prob,prob_n-prob,prob_n,fc_n,n1_n,n2_n,pc_n);
  alpha=min(prob_n-prob,0);
  if (alpha>0){
    n1=n1_n;
    n2=n2_n;
    pc=pc_n;
    fc=fc_n;
    prob=prob_n;
    counter++;}
  else{
    rnd=log(uniform(0.0,1.0));
    if(rnd<alpha){
      n1=n1_n;
      n2=n2_n;
      pc=pc_n;
      fc=fc_n;
      prob=prob_n;
      counter++;}
    }}while(counter<=steps);
  return prob;
}


int main(int argc, char* argv[]) {
srand(time(0));

/* the arguments of the function I need are the  following:
n1,n2,pc,pmin,pmax,delta
*/
char *name;
int N, Nres, Nbins;
double L, sv, st;


name=argv[1];
sscanf(argv[2],"%lf",&L);
sscanf(argv[3],"%d",&N);
sscanf(argv[4],"%d",&Nres);
sscanf(argv[5],"%lf",&sv);
sscanf(argv[6],"%lf",&st);
Nbins=argc-7;
double pmin,pmax,dx,pc;

pmin=4.0/L;
pc=64.0/L;
pmax=(1.0*N-2.0)/(4.0*L);
dx=L/(1.0*N);

float delta[Nbins];
for(int i = 0; i < Nbins; i++){
 sscanf(argv[7+i],"%f",&delta[i]);}

double n1_min,n1_max,n2_min,n2_max,f_min,f_max;
n1_min=0.0;
n1_max=5.0;
n2_min=1.0;
n2_max=8.0;
f_min=0.1;
f_max=100.0;
char* s = concat("Temp_Files/", name);
s       = concat(s,"_tabla_res.txt");
double **circulation;
circulation=two_column_file(s,Nres);
free(s);

double temp;
temp=walker(n1_min, n1_max, n2_min, n2_max, f_min,
  f_max,  pmin,  pmax,  Nbins, delta, st, sv, circulation, Nres,50,dx);
printf("%f\n", temp);
}
