import os, sys, getopt
from shared_resources_planning import SharedResourcesPlanning


# ======================================================================================================================
#  Read Execution Arguments
# ======================================================================================================================
def print_help_message():
    print('\nShared Resources Planning Tool usage:')
    print('\n Usage: main.py [OPTIONS]')
    print('\n Options:')
    print('   {:25}  {}'.format('-d, --test_case=', 'Directory of the Test Case to be run (located inside "data" directory)'))
    print('   {:25}  {}'.format('-f, --specification_file=', 'Specification file of the Test Case to be run (located inside the "test_case" directory)'))
    print('   {:25}  {}'.format('-h, --help', 'Help. Displays this message'))


def read_execution_arguments(argv):

    test_case_dir = str()
    spec_filename = str()

    try:
        opts, args = getopt.getopt(argv, 'hd:f:', ['help', 'test_case=', 'specification_file='])
    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)

    if not argv or not opts:
        print_help_message()
        sys.exit()

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help_message()
            sys.exit()
        elif opt in ('-d', '--dir'):
            test_case_dir = arg
        elif opt in ('-f', '--file'):
            spec_filename = arg

    if not test_case_dir or not spec_filename:
        print('Shared Resource Planning Tool usage:')
        print('\tmain.py -d <test_case> -f <specification_file>')
        sys.exit()

    return spec_filename, test_case_dir


# ======================================================================================================================
#  Shared Resources Planning
# ======================================================================================================================
def shared_resources_planning(working_directory, specification_filename):

    print('==========================================================================================================')
    print('                                     ATTEST -- SHARED RESOURCES PLANNING                                  ')
    print('==========================================================================================================')

    planning_problem = SharedResourcesPlanning(working_directory, specification_filename)
    planning_problem.read_planning_problem()
    planning_problem.run_planning_problem()

    print('==========================================================================================================')
    print('                                               END                                                        ')
    print('==========================================================================================================')


# ======================================================================================================================
#  Main
# ============================================================================cal==========================================
if __name__ == '__main__':

    filename, test_case = read_execution_arguments(sys.argv[1:])
    directory = os.path.join(os.getcwd(), 'data', test_case)
    shared_resources_planning(directory, filename)

    '''
    filename = 'HR1.txt'
    directory = os.path.join(os.getcwd(), 'data', 'HR')
    shared_resources_planning(directory, filename)
    '''




