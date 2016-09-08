
int okreni_90 (int broj, int polja){
    int resenje=0;int x[8];int y[8];
    
    for (int j=polja; j > 0; j--){x[j] = broj%10; broj /= 10;}   //rasclanjivanje broja na elemente
    for (int j=1; j<=polja; j++){y[polja-x[j]+1]=j;}             //preracun za okretanje za 90 stepeni

    for (int g=1; g<=polja; g++){resenje += y[g]*pow(10, g-1);}
    //printf("resenje=%u\n", resenje);  
    return resenje;       
}
int okreni180 (int broj, int polja){
    int resenje=0;int x[8];int y[8];
    
    for (int j=polja; j > 0; j--){x[j] = broj%10; broj /= 10;}   //rasclanjivanje broja na elemente
    //for(int j=1; j<=polja; j++){y[polja-x[j]+1]=j;}              //preracun za okretanje za 180 stepeni
    for(int j=1; j<=polja; j++){y[j]=x[j];}
    for(int j=1; j<=polja; j++){x[j]=y[polja-j+1];}
    
    for (int g=1; g<=polja; g++){resenje += x[g]*pow(10, g-1);}
    return resenje;       
}

// Formirano je 4,096 work-group sa po 256 work-items (4,096*256=1,048,576 tj. 2na20)
// work-group velicina =256 (work-items). Definisano u clojure. (enq-nd! cqueue reduction-scalar (work-size [num-items] [workgroup-size (=256 mi smo definisali)]) nil profile-event)
__kernel void reduction_scalar(__global int* data,
                              __global int* brpolja,
                              __local int4* partial_sums,         //resenja
                              __local int* partial_podaci,        //lokalni podaci preneti iz globalne memorije
                              __local int* x,
                             // __local int* y,                                                            
                              __global int4* output) {   
   
   int tr,j,g,f;        
   int gid = get_global_id(0);
   int lid = get_local_id(0);
   int gsize = get_local_size(0);
       
   partial_podaci[lid] = data[gid];
   int polja=brpolja[0];
   barrier(CLK_LOCAL_MEM_FENCE); 

                
   for (int k=0; k<=gsize; k++){        
        //j=polja; while (tr > 0){x[j] = tr%10; tr /= 10; j--;}      
        tr=partial_podaci[k]; for (int j=polja; j > 0; j--){x[j] = tr%10; tr /= 10;}    //rasclanjivanje broja na elemente
         
        int bre=1;int i=0;
        while (i < polja && bre==1){
            for(j=i+1; j<=polja; j++){
                if (!((x[i]==x[j]) || (abs(x[i]-x[j])==abs(i-j)))){
                    if (j == polja && i == polja-1) {                    //resenje
                       partial_sums[lid].s0 = partial_podaci[k];                       //tabla osnovna sa resenjem bez okretanja                                          
                       partial_sums[lid].s1 = okreni_90(partial_sums[lid].s0, polja);  //okrecemo tablu za 90 stepeni.  0->90
                       partial_sums[lid].s2 = okreni180(partial_sums[lid].s0, polja);  //okrecemo tablu za 180 stepeni. 0->180
                       partial_sums[lid].s3 = okreni180(partial_sums[lid].s1, polja);  //okrecemo tablu za 180 stepeni. 90->270                                                                                                                               
                    }                  
                }
                else {bre=0;}
            }
        i++;}
        barrier(CLK_LOCAL_MEM_FENCE);     
   } 
   //if(lid == 0) {output[get_group_id(0)] = partial_sums[lid];}
   if(lid == 0) {output[get_group_id(0)].s0 = partial_sums[lid].s0;
                 output[get_group_id(0)].s1 = partial_sums[lid].s1;
                 output[get_group_id(0)].s2 = partial_sums[lid].s2;
                 output[get_group_id(0)].s3 = partial_sums[lid].s3;}      
}
__kernel void reduction_scalar3(__global int* data,
                              __global int* brpolja,
                              __local int4* partial_sums,         //resenja
                              __local int* partial_podaci,        //lokalni podaci preneti iz globalne memorije
                              __local int* x,
                              __local int* y,                                                            
                              __global int4* output) {   
   
   int tr,j,g,f;        
   int gid = get_global_id(0);
   int lid = get_local_id(0);
   int gsize = get_local_size(0);
       
   partial_podaci[lid] = data[gid];
   int polja=brpolja[0];
   //printf("gid=%u lid=%u get_group_id(0)=%u partial_podaci[lid]=%u data[gid]=%u\n",  gid, lid, get_group_id(0), partial_podaci[lid], data[gid]);
   barrier(CLK_LOCAL_MEM_FENCE); 

                
   for (int k=0; k<=gsize; k++){        
        //j=polja; while (tr > 0){x[j] = tr%10; tr /= 10; j--;}      
        tr=partial_podaci[k]; for (int j=polja; j > 0; j--){x[j] = tr%10; tr /= 10;}    //rasclanjivanje broja na elemente
         
        int bre=1;int i=0;
        while (i < polja && bre==1){
            for(j=i+1; j<=polja; j++){
                if (!((x[i]==x[j]) || (abs(x[i]-x[j])==abs(i-j)))){
                    if (j == polja && i == polja-1) {                    //resenje
                       partial_sums[lid].s0 = partial_podaci[k];                       //tabla osnovna sa resenjem bez okretanja                                          
                       partial_sums[lid].s1 = okreni_90(partial_sums[lid].s0, polja);  //okrecemo tablu za 90 stepeni.  0->90
                       partial_sums[lid].s2 = okreni180(partial_sums[lid].s0, polja);  //okrecemo tablu za 180 stepeni. 0->180
                       partial_sums[lid].s3 = okreni180(partial_sums[lid].s1, polja);  //okrecemo tablu za 180 stepeni. 90->270                                                                                                                               
                    }                  
                }
                else {bre=0;}
            }
        i++;}
        barrier(CLK_LOCAL_MEM_FENCE);     
   } 
   //if(lid == 0) {output[get_group_id(0)] = partial_sums[lid];}
   if(lid == 0) {output[get_group_id(0)].s0 = partial_sums[lid].s0;
                 output[get_group_id(0)].s1 = partial_sums[lid].s1;
                 output[get_group_id(0)].s2 = partial_sums[lid].s2;
                 output[get_group_id(0)].s3 = partial_sums[lid].s3;                                  
   }     
}
 
__kernel void reduction_scalar2(__global int* data,
                                __global int* brpolja,
                                __local int* partial_sums,          //resenja
                                __local int* partial_podaci,        //lokalni podaci preneti iz globalne memorije
                                __local int* x,                              
                                __global int* output) {   
   
   int trenutno,j;        
   int gid = get_global_id(0);
   int lid = get_local_id(0);
   int gsize = get_local_size(0);
   
   partial_podaci[lid] = data[gid];      
   int polja=brpolja[0];
   barrier(CLK_LOCAL_MEM_FENCE);
    
   //printf("gid=%u lid=%u get_group_id(0)=%u partial_podaci[lid]=%u data[gid]=%u\n",  gid, lid, get_group_id(0), partial_podaci[lid], data[gid]); 
   
   int koji=0;int s=0;             
   for (int k=0; k<gsize; k++){
        trenutno=partial_podaci[k];   
        j=polja; while (trenutno > 0){x[j] = trenutno%10; trenutno /= 10; j--;}      //rasclanjivanje broja na elemente
                        //printf("lid=%u get_group_id(0)=%u k==%u partial_podaci[k]=%u\n",  lid, get_group_id(0), k, partial_podaci[lid]);    
        int bre=1;int i=0;  
        //printf("partial_podaci[k]=%u\n",  partial_podaci[k]);
        while (i < polja && bre==1){
            for(j=i+1; j<=polja; j++){
                if (!((x[i]==x[j]) || (abs(x[i]-x[j])==abs(i-j)))){
                    if (j == polja && i == polja-1) {
                        partial_sums[lid] = partial_podaci[k];
                        //if (s==0){output[get_group_id(0)].s0 = partial_podaci[k]; s=1;}
                        //else {output[get_group_id(0)].s1 = partial_podaci[k];s=0;}
     //switch(s)
     //{
     //    case 0:
     //      output[get_group_id(0)].s0 = partial_podaci[k]; s=1;
     //      break;
     //    case 1:
     //      output[get_group_id(0)].s1 = partial_podaci[k]; s=2;
     //      break;
     //    case 2:
     //      output[get_group_id(0)].s2 = partial_podaci[k]; s=3;
     //      break;
     //    case 3:
     //      output[get_group_id(0)].s3 = partial_podaci[k]; s=4;
     //      break;
     //    case 4:
     //      output[get_group_id(0)].s4 = partial_podaci[k]; s=5;
     //      break;
     //    case 5:
     //      output[get_group_id(0)].s5 = partial_podaci[k]; s=6;
     //      break; 
     //    case 6:
     //      output[get_group_id(0)].s6 = partial_podaci[k]; s=7;
     //      break;                                             
     //    case 7:
     //      output[get_group_id(0)].s7 = partial_podaci[k]; s=0;
     //      break;                   
    //}                                                                        
                        //printf("gid=%u lid=%u get_group_id(0)=%u k==%u partial_podaci[k]=%u\n",  gid, lid, get_group_id(0), k, partial_podaci[k]);                                                                                                  
                    }
                }
                else {bre=0;}
            }
        i++;}
        barrier(CLK_LOCAL_MEM_FENCE);     
   } 
    
   if(lid == 0) {output[get_group_id(0)] = partial_sums[lid];}        //Vraca vrednost pojedinacne work-group (Na kraju krajeva za svih 1,024 komada)   
   //  if(lid == 0) {output[get_group_id(0)] = partial_podaci[lid];}
}
