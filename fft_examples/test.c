#include<stdio.h>
#define DATASIZE 8

int main(void){
int n[] = { DATASIZE };
int a = DATASIZE;
printf("n[] %lu\n", sizeof(n));
printf("n[] %lu\n", sizeof(a));
return 0;
}
