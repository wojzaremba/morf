#include <stdio.h>
#include <math.h>

static int lookups = 0;

template<class TYPE> TYPE* at(TYPE* name, int idx) {
	lookups++;
	return name + idx;
}

#if defined(DEBUG)
	#define AT(name, idx) (*(at(name, idx)))
	#define print(format, args...) mexPrintf(format, args);
#else
	#define AT(name, idx) (*(name + idx))
	#define print(format, args...) 
#endif

