# extern THCState *state;
THCState *state = at::globalContext().lazyInitCUDA();
