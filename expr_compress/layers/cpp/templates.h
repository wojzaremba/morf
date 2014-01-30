#include <stdio.h>
#include <math.h>

static int lookups = 0;

template<class TYPE> TYPE* at(TYPE* name, int idx) {
	lookups++;
	return name + idx;
}

#define assert_(EXP) { if (!(bool)(EXP)) { \
                       mexPrintf("FILE : %s, LINE : %d, EXP: %s \n", __FILE__, __LINE__, #EXP); \
                       mexErrMsgTxt("!!!! error\n"); \
                     } } 

#if defined(DEBUG)
	#define AT(name, idx) (*(at(name, idx)))
	#define print(format, args...) mexPrintf(format, args);
#else
	#define AT(name, idx) (*(name + idx))
	#define print(format, args...) 
#endif

