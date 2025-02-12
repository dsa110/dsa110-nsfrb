#define PY_SSIZE_T_CLEAN
#include </home/ubuntu/msherman_nsfrb/miniconda/pkgs/python-3.10.9-he550d4f_0_cpython/include/python3.10/Python.h>
#include <sys/mman.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/*
 * This module is a C-implemented Python module which reads data from a stream and outputs it as a
 * numpy array. The express purpose is to read fast visibility data output from the dsax-nsfrb.c 
 * code and sending it to the realtime imager. This is based on a tutorial from 
 * https://docs.python.org/3/extending/extending.html
 */

static PyObject *rtreader_read(PyObject *self, PyObject *args)
{
	// parsing Python arguments
	//FILE *fptr;
	const char *address;
	//int *fdptr;
	//int sts;

	if (!PyArg_ParseTuple(args, "s", &address))
		return NULL;
	long int address_int = strtol(address,NULL,0);
	//int fd = fdptr[0];
	printf("Pointer Address: %ld\n",address_int);
	// printf("File decriptor: %d\n",fd);

	//cast to pointer
	char *memaddr = (char *)address_int;
	//FILE *fptr = (FILE *)address_int;
	printf("Memory Address: %p\n",memaddr);

	//make memory map
	int PROT_FLAG = PROT_READ;
        int MAP_FLAG = MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED;
	size_t buffersize = 11;
	memaddr = mmap(memaddr, buffersize, PROT_FLAG, MAP_FLAG, -1, 0);

	//copy from memory
        char buffer2[buffersize];
	memcpy(buffer2,memaddr,buffersize);

	//open and read
	//size_t buffersize = 11;
	//char buffer2[buffersize];
	//fread(buffer2,1,buffersize,fptr);
	printf("Contents: %s\n",buffer2);
	//fclose(fptr);

	//sts = system(command);
	// printf("%s",command);
	return PyLong_FromLong(address_int);
}

//Method table
static PyMethodDef rtreaderMethods[] = {
	{"read", rtreader_read, METH_VARARGS,
	 "Read from the given address."},
	{NULL, NULL, 0, NULL} /* Sentinel */
};

//Module definition
static struct PyModuleDef rtreadermodule = {
	PyModuleDef_HEAD_INIT,
	"rtreader", /* name of module */
	NULL, /*module documentation */
	-1, /*size of per-interpreter state of module or -1 if module keeps state in global vars*/
	rtreaderMethods
};

//Initialization function
PyMODINIT_FUNC PyInit_rtreader(void)
{
	return PyModule_Create(&rtreadermodule);
}

//add to PyImport_Inittab table
int main(int argc, char *argv[])
{
        PyStatus status;
        PyConfig config;
	PyConfig_InitPythonConfig(&config);

	/* Add a built-in module, before Py_Initialize */
	if (PyImport_AppendInittab("rtreader", PyInit_rtreader) == -1) {
		fprintf(stderr, "Error: could not extend in-built modules table\n");
		exit(1);
	}
	
	/* Pass argv[0] to the Python interpreter */
	status = PyConfig_SetBytesString(&config, &config.program_name, argv[0]);
	if (PyStatus_Exception(status)) {
		goto exception;
	}

	/* Initialize the Python interpreter.  Required.
	 * If this step fails, it will be a fatal error. */
	status = Py_InitializeFromConfig(&config);
	if (PyStatus_Exception(status)) {
		goto exception;
	}
	PyConfig_Clear(&config);

	return 0;

	exception:
		PyConfig_Clear(&config);
		Py_ExitStatusException(status);
}
