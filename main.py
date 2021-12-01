import time
import chess
import numpy as np
import chess.polyglot
import chess.syzygy
import chess.svg
import chess.pgn
import tensorflow as tf
from IPython.display import display, HTML, clear_output
import random
import numpy
import sys
import os
import multiprocessing
import itertools
from itertools import chain

i = 0
ind = 0
full_matrix = []
eval_matrix = []
eval_matrix1 = []
board = chess.Board()
squares_list = [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1,
                chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2,
                chess.A3, chess.B3, chess.C3, chess.D3, chess.E3, chess.F3, chess.G3, chess.H3,
                chess.A4, chess.B4, chess.C4, chess.D4, chess.E4, chess.F4, chess.G4, chess.H4,
                chess.A5, chess.B5, chess.C5, chess.D5, chess.E5, chess.F5, chess.G5, chess.H5,
                chess.A6, chess.B6, chess.C6, chess.D6, chess.E6, chess.F6, chess.G6, chess.H6,
                chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7,
                chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8]


def white_attacked(board):
    for i in range(64):
        if board.is_attacked_by(chess.BLACK, squares_list[i - 1]):
            return True
        else:
            i = i


def black_attacked(board):
    for i in range(64):
        if board.is_attacked_by(chess.WHITE, squares_list[i - 1]):
            return True
        else:
            i = i


def get_move(prompt):
    uci = input(prompt)
    if uci and uci[0] == "q":
        raise KeyboardInterrupt()
    try:
        chess.Move.from_uci(uci)
    except:
        uci = None
    return uci


def staticAnalysis(board, move, my_color):
    score = random.random()
    ## Check some things about this move:
    board.push(move)
    # Now check some other things:
    for (piece, value) in [(chess.PAWN, 1),
                           (chess.BISHOP, 3),
                           (chess.QUEEN, 9),
                           (chess.KNIGHT, 3),
                           (chess.ROOK, 5)]:
        score += len(board.pieces(piece, my_color)) * value
        score -= len(board.pieces(piece, not my_color)) * value
        # can also check things about the pieces position here
        score += 100 if board.is_checkmate() else 0
        score += 1 if board.is_check() else 0
        score += 3 if board.is_capture(move) else 0
        score += 1 if black_attacked(board) else 0
        score -= 1 if white_attacked(board) else 0
    return score


def human_player(board):
    display(board)
    uci = get_move("%s's move [q to quit]> " % who(board.turn))
    legal_uci_moves = [move.uci() for move in board.legal_moves]
    while uci not in legal_uci_moves:
        print("Legal moves: " + (",".join(sorted(legal_uci_moves))))
        uci = get_move("%s's move[q to quit]> " % who(board.turn))
    return uci


def player1(board):
    moves = list(board.legal_moves)
    for move in moves:
        newboard = board.copy()
        move.score = staticAnalysis(newboard, move, board.turn)
    moves.sort(key=lambda move: move.score, reverse=True)  # sort on score
    return moves[0].uci()


def player2(board):
    move = minimaxRoot(5, board, True)
    move = chess.Move.from_uci(str(move))
    return move


def who(player):
    return "White" if player == chess.WHITE else "Black"


def display_board(board, use_svg):
    if use_svg:
        return board._repr_svg_()
    else:
        return "<pre>" + str(board) + "</pre>"


def play_game(player1, player2, visual="svg", pause=0.1):
    global i
    use_svg = (visual == "svg")
    board = chess.Board()
    try:
        while not board.is_game_over(claim_draw=True):
            if i >= 74:
                break
            if board.turn == chess.WHITE:
                uci = player1(board)
            else:
                uci = player2(board)
            name = who(board.turn)
            board.push_uci(uci)
            board_stop = display_board(board, use_svg)
            html = "<b>Move %s %s, Play '%s':</b><br/>%s" % (len(board.move_stack), name, uci, board_stop)
            if visual is not None:
                if visual == "svg":
                    clear_output(wait=True)
                display(HTML(html))
                if visual == "svg":
                    time.sleep(pause)
    except KeyboardInterrupt:
        msg = "Game interrupted!"
        return (None, msg, board)
    result = None
    if board.is_checkmate():
        msg = "checkmate: " + who(not board.turn) + " wins!"
        result = not board.turn
    elif board.is_stalemate():
        msg = "draw: stalemate"
    elif board.is_fivefold_repetition():
        msg = "draw: 5-fold repetition"
    elif board.is_insufficient_material():
        msg = "draw: insufficient material"
    elif board.can_claim_draw():
        msg = "draw: claim"
    else:
        msg = "Resigned"
    if visual is not None:
        print(msg)
    return (result, msg, board)


def load_data():
    inputs = open("/Users/kennyhermus/Downloads/ficsgamesdb_201811_chess2000_nomovetimes_115863.pgn")
    while True:
        try:
            game = chess.pgn.read_game(inputs)
            global inputs_matrix
            inputs_matrix = list(game.mainline_moves())
            return inputs_matrix
        except KeyboardInterrupt:
            raise
        except:
            continue
        if not game:
            break
        return inputs_matrix


def gm_player(board):
    global i
    global inputs_matrix
    load_data()
    i += 1
    return inputs_matrix[i - 1].uci()


def random_player(board):
    move = random.choice(list(board.legal_moves))
    print(list(board.legal_moves))
    return move.uci()


def white_wins():
    training_inputs = open("/Users/kennyhermus/Downloads/ficsgamesdb_201811_chess2000_nomovetimes_115863.pgn")
    global white_wins
    white_wins = []
    global ind
    for offset_game, headers in chess.pgn.scan_headers(training_inputs):
        if headers["Result"] == "1-0":
            game = chess.pgn.read_game(training_inputs)
            white_wins[ind] = list(game.mainline_moves())
            ind
    return white_wins


def white_moves():
    r = 1
    global white_wins
    for r in white_wins:
        del white_wins[r]
        r += 2
    return white_wins


def chess_model():
    model = keras.Sequential([
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(full_matrix, eval_matrix, verbose=2)


import chess
import math
import random
import sys


def minimaxRoot(depth, board, isMaximizing):
    possibleMoves = board.legal_moves
    bestMove = -9999
    bestMoveFinal = None
    for x in possibleMoves:
        move = chess.Move.from_uci(str(x))
        board.push(move)
        value = max(bestMove, minimax(depth - 1, board, -10000, 10000, not isMaximizing))
        board.pop()
        if (value > bestMove):
            print("Best score: ", str(bestMove))
            print("Best move: ", str(bestMoveFinal))
            bestMove = value
            bestMoveFinal = move
    return bestMoveFinal


def minimax(depth, board, alpha, beta, is_maximizing):
    if (depth == 0):
        return -evaluation(board)
    possibleMoves = board.legal_moves
    if (is_maximizing):
        bestMove = -9999
        for x in possibleMoves:
            move = chess.Move.from_uci(str(x))
            board.push(move)
            bestMove = max(bestMove, minimax(depth - 1, board, alpha, beta, not is_maximizing))
            board.pop()
            alpha = max(alpha, bestMove)
            if beta <= alpha:
                return bestMove
        return bestMove
    else:
        bestMove = 9999
        for x in possibleMoves:
            move = chess.Move.from_uci(str(x))
            board.push(move)
            bestMove = min(bestMove, minimax(depth - 1, board, alpha, beta, not is_maximizing))
            board.pop()
            beta = min(beta, bestMove)
            if (beta <= alpha):
                return bestMove
        return bestMove


def calculateMove(board):
    possible_moves = board.legal_moves
    if (len(possible_moves) == 0):
        print("No more possible moves...Game Over")
        sys.exit()
    bestMove = None
    bestValue = -9999
    n = 0
    for x in possible_moves:
        move = chess.Move.from_uci(str(x))
        board.push(move)
        boardValue = -evaluation(board)
        board.pop()
        if (boardValue > bestValue):
            bestValue = boardValue
            bestMove = move
    return bestMove


def evaluation(board):
    i = 0
    evaluation = 0
    x = True
    try:
        x = bool(board.piece_at(i).color)
    except AttributeError as e:
        x = x
    while i < 63:
        i += 1
        evaluation = evaluation + (
            getPieceValue(str(board.piece_at(i))) if x else -getPieceValue(str(board.piece_at(i))))
    return evaluation


def getPieceValue(piece):
    if (piece == None):
        return 0
    value = 0
    if piece == "P" or piece == "p":
        value = 10
    if piece == "N" or piece == "n":
        value = 30
    if piece == "B" or piece == "b":
        value = 30
    if piece == "R" or piece == "r":
        value = 50
    if piece == "Q" or piece == "q":
        value = 90
    if piece == 'K' or piece == 'k':
        value = 900
    # value = value if (board.piece_at(place)).color else -value
    return value


def main():
    board = chess.Board()
    n = 0
    print(board)
    while n < 100:
        if n % 2 == 0:
            move = input("Enter move: ")
            move = chess.Move.from_uci(str(move))
            board.push(move)
        else:
            print("Computers Turn:")
            move = minimaxRoot(5, board, True)
            move = chess.Move.from_uci(str(move))
            board.push(move)
        print(board)
        n += 1


def read_all():
    inputs = open("/Users/kennyhermus/Downloads/ficsgamesdb_201811_chess2000_nomovetimes_115863.pgn")
    global inputs_matrix
    global full_matrix
    global full_matrix1
    for j in range(2):
        game = chess.pgn.read_game(inputs)
        inputs_matrix = list(game.mainline_moves())
        full_matrix.append(inputs_matrix)
    full_matrix1 = list(chain.from_iterable(full_matrix))
    return full_matrix


def eval_all():
    board = chess.Board()
    read_all()
    global eval_matrix
    global eval_matrix1
    global full_matrix
    ind = 0
    ind1 = 0
    for ind in range(len(full_matrix)):
        for ind1 in range(len(full_matrix[ind])):
            move = chess.Move.from_uci(str(full_matrix[ind][ind1]))
            board.push(move)
            eval_matrix1.append(evaluation(board))
        eval_matrix.append(eval_matrix1)
    return eval_matrix


# play_game(player2, player2)
# load_data()
main()
# load_data()
# read_all()
# eval_all()