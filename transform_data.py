def transform(file_in):
    """
    The main function for the program to transfrom from the given data to
    more data based on the rotation of the 2048-board
    """
    # Prepare file output
    file_out = open("only_2048_3_transformed.csv", "w")

    # Each line the file_in is a 2048 board and the move of AI for that board
    # Using strip to remove th \n at the end of line
    line = file_in.readline().strip()
    # Convert the board to a list to access element easier
    while len(line) != 0:
        current_board = line.split(",")
        board1 = rotate_90degree_anticlockwise(current_board)
        board2 = rotate_90degree_anticlockwise(board1)
        board3 = rotate_90degree_anticlockwise(board2)

        write_board_to_file(current_board, file_out)
        write_board_to_file(board1, file_out)
        write_board_to_file(board2, file_out)
        write_board_to_file(board3, file_out)

        line = file_in.readline().strip()


def rotate_90degree_anticlockwise(current_board):
    """
    Function to rotate the input board 90 degree, anticlockwise

    return: a new board after roration
    """
    # Get each of row of the board to rotate easier
    first_row = current_board[:4]
    second_row = current_board[4:8]
    third_row = current_board[8:12]
    forth_row = current_board[12:16]
    current_move = current_board[-1]

    list_rows = [first_row, second_row, third_row, forth_row]
    # A rotation of 90 degree anticlockwise will make the direction change:
    #   up -> left (0 - 2)
    #   down -> right (1 - 3)
    #   left -> down (2 - 1)
    #   right -> up (3 - 0)
    dict_replace_direction = {"0": "2", "1": "3", "2": "1", "3": "0"}

    new_board = []

    # The last element of each old_row become the first element of each new_row
    #   second last element becomes the second element
    #       so on
    for index in range(3, -1, -1):
        for row in list_rows:
            new_board.append(row[index])

    new_move = dict_replace_direction[current_move]

    new_board.append(new_move)
    return new_board


def write_board_to_file(board, file_out):
    """
    Method to write the input board to the file_out
    """
    for index in range(len(board) - 1):
        file_out.write(board[index] + ",")
    # Write the move to the end of line
    file_out.write(board[-1] + "\n")


file_in = open("./only_2048_3.csv", "r")
transform(file_in)
