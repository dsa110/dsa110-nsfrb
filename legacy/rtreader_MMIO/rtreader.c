#define PY_SSIZE_T_CLEAN
#include </home/ubuntu/msherman_nsfrb/miniconda/pkgs/python-3.10.9-he550d4f_0_cpython/include/python3.10/Python.h>
#include <sys/mman.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <sys/shm.h>

/*
 * This module is a C-implemented Python module which reads data from a stream and outputs it as a
 * numpy array. The express purpose is to read fast visibility data output from the dsax-nsfrb.c 
 * code and sending it to the realtime imager. This is based on a tutorial from 
 * https://docs.python.org/3/extending/extending.html
 */

static PyObject *rtreader_read(PyObject *self, PyObject *args)
{
	// parsing Python arguments
	//const char *shmid_str;
	long shmid;
	long buffersize;
	
	//if (!PyArg_ParseTuple(args, "s", &shmid_str))
	//	return NULL;
	//int shmid = atoi(shmid_str);
	if (!PyArg_ParseTuple(args,"ll", &shmid, &buffersize))
		return NULL;
	//printf("SHMID: %ld\n",shmid);
	//printf("BUFFERSIZE: %ld\n",buffersize);

	//printf("%p\n",shmid_ptr);
	//printf("%d\n",shmid_ptr);
	//int shmid = *shmid_ptr;
	//fflush(stdout);

	//attach
	char *memaddr = shmat(shmid, NULL, SHM_RDONLY);
	
	//copy to buffer
	//size_t buffersize = sizeof(memaddr)/sizeof(char);
	char buffer[buffersize];
	memcpy(buffer,memaddr,buffersize);

	//print
	//printf("Contents: %s\n",memaddr);
	//printf("Copied: %s\n",buffer);
	//detach
        shmdt(memaddr);
	shmctl(shmid,IPC_RMID,NULL);

	
	
	return PyByteArray_FromStringAndSize(buffer,buffersize);//PyBytes_FromString(buffer); //PyLong_FromLong(shmid);
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
