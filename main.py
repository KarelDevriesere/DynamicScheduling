import sys
import os
import random
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import gcol
import csv

import seaborn as sns
import pandas as pd
from scipy import stats

import gurobipy as gp
from gurobipy import GRB

real_lambda_h_CL = [1.33, 4.0, 1.67, 1.67, 1.67, 2.33, 0.67, 0.33, 1.0, 0.67, 0.33, 2.0, 3.0, 1.0, 0.33, 1.0, 1.67, 1.0, 2.0, 2.33, 1.33, 2.33, 1.33, 2.33, 1.33, 2.0, 2.33, 0.33, 0.67, 0.0, 0.67, 1.67, 1.33, 1.67, 1.33, 0.67, 1.0, 2.67, 3.0, 2.0, 2.33, 1.67, 1.33, 1.67, 1.67, 0.67, 1.0, 1.0, 3.0, 1.33, 1.67, 1.0, 1.67, 0.33, 2.0, 2.0, 1.0, 1.0, 1.33, 2.67, 1.0, 1.67, 0.67, 1.0, 1.67, 2.33, 0.67, 3.0, 2.67, 1.0, 0.67, 2.0, 0.67, 3.0, 4.0, 0.33, 2.0, 1.0, 1.67, 1.33, 1.0, 0.33, 3.0, 1.67, 1.0, 4.0, 1.33, 2.0, 1.67, 1.33, 1.0, 0.67, 2.33, 1.67, 1.0, 0.67, 2.0, 2.33, 1.67, 1.67, 1.67, 1.33, 1.0, 2.33, 1.67, 0.67, 1.0, 1.0, 1.0, 0.67, 1.0, 0.67, 2.0, 2.0, 0.67, 0.67, 1.0, 1.67, 2.0, 1.0, 1.33, 1.67, 2.33, 0.67, 0.33, 1.67, 1.0, 2.67, 1.67, 1.67, 1.0, 2.67, 1.33, 2.67, 1.33, 0.33, 1.33, 0.67, 2.0, 1.0, 1.67, 1.0, 1.67, 0.67, 1.33, 1.67, 1.0, 0.33, 2.67, 2.0, 2.33, 0.33, 0.67, 1.33, 0.0, 1.67, 1.0, 0.67, 0.33, 1.33, 0.67, 2.67, 2.67, 0.67, 2.67, 1.33, 1.33, 1.0, 1.67, 1.33, 4.67, 0.33, 1.0, 2.0, 0.67, 1.0, 0.67, 3.0, 3.0, 0.67, 2.0, 3.33, 1.67, 3.0, 2.33, 2.67, 1.67, 0.0, 1.67, 1.33, 0.33, 0.67, 1.67, 2.67, 1.33, 4.33, 1.33, 2.33, 0.0, 1.67, 1.0, 1.67, 0.67, 0.67, 0.33, 0.67, 1.0, 0.67, 3.33, 3.67, 1.0, 1.33, 1.33, 1.0, 2.67, 0.33, 2.0, 1.33, 1.0, 2.0, 1.33, 1.0, 0.67, 2.0, 1.67, 1.33, 1.33, 2.67, 1.33, 1.67, 4.0, 1.0, 0.33, 2.33, 3.67, 1.0, 1.33, 1.33, 0.67, 0.67, 1.33, 1.33, 3.0, 1.67, 3.33, 2.33, 2.0, 1.67, 0.33, 1.67, 2.0, 0.67, 1.0, 1.67, 0.33, 0.67, 3.0, 0.67, 2.33, 1.67, 4.33, 0.67, 1.67, 2.33, 1.67, 0.33, 1.0, 1.33, 1.0, 2.0, 1.67, 0.67, 1.33, 0.67, 3.33, 3.33, 1.67, 2.0, 2.0, 1.67, 0.67, 3.33, 0.0, 2.0, 1.67, 0.67, 0.67, 1.67, 2.33, 2.33, 2.0, 1.0, 2.0, 1.67, 3.33, 2.33, 3.0, 2.33, 0.67, 1.67, 1.67, 0.33, 0.67, 0.67, 1.67, 3.33, 3.33, 1.33, 0.33, 1.67, 1.0, 1.67, 1.0, 0.67, 2.33, 1.0, 0.67, 3.0, 2.67, 0.67, 3.0, 3.67, 2.67, 1.33, 0.67, 4.67, 0.67, 2.33, 1.33, 2.33, 1.33, 0.33, 1.33, 1.33, 1.0, 0.33, 1.33, 2.33, 1.67, 1.0, 1.67, 2.33, 2.33, 2.0, 2.0, 1.0, 4.33, 0.33, 2.0, 1.33, 1.0, 1.33, 3.0, 3.67, 1.0, 0.0, 1.67, 2.0, 1.0, 2.0, 2.33, 1.33, 1.33, 0.67, 0.33, 1.33, 1.33, 1.33, 3.33, 1.67, 5.0, 1.0, 1.0, 2.0, 1.0, 3.33, 2.0, 1.0, 2.67, 1.0, 2.33, 1.67, 0.33, 1.67, 2.33, 1.0, 2.33, 1.67, 4.0, 2.33, 1.0, 2.67, 3.0, 0.67, 1.67, 1.33, 3.0, 0.33, 0.67, 2.67, 3.0, 5.0, 3.33, 1.33, 2.67, 1.33, 2.33, 1.0, 0.67, 1.33, 0.33, 2.33, 1.33, 0.33, 0.33, 0.33, 2.33, 2.67, 1.0, 1.67, 3.0, 2.0, 1.67, 1.0, 1.67, 0.33, 1.0, 2.0, 1.33, 2.67, 0.67, 0.67, 2.33, 1.67, 2.67, 1.67, 1.0, 3.33, 0.33, 2.33, 1.33, 1.33, 0.33, 2.33, 1.67, 2.67, 1.33, 1.0, 2.67, 3.67, 2.67, 2.67, 1.0, 2.0, 0.67, 1.33, 0.33, 3.33, 3.0, 0.33, 1.67, 0.33, 1.67, 1.0, 2.67, 2.33, 1.33, 1.67, 1.0, 2.0, 2.0, 1.33, 1.67, 0.67, 2.0, 1.67, 2.0, 0.67, 2.33, 0.67, 3.0, 1.0, 2.33, 2.67, 1.33, 1.33, 3.0, 0.67, 0.67, 0.67, 1.0, 1.0, 0.67, 1.0, 0.33, 1.67, 2.33, 1.67, 2.33, 1.67, 2.33, 3.33, 1.0, 2.0, 1.33, 0.67, 2.67, 1.33, 1.0, 1.33, 0.33, 1.0, 1.67, 1.33, 4.33, 0.33, 2.33, 3.0, 2.0, 1.33, 2.0, 2.0, 1.33, 0.33, 1.67, 0.33, 2.0, 0.33, 1.67, 1.67, 1.0, 2.33, 1.67, 4.0, 3.33, 2.67, 1.0, 0.67, 3.0, 0.67, 2.33, 1.0, 0.33, 1.0, 3.33, 1.33, 3.0, 0.33, 1.67, 3.33, 2.33, 1.0, 2.0, 0.67, 0.67, 2.33, 1.33, 2.33, 0.33, 1.33, 3.0, 1.67, 2.0, 3.67, 1.33, 1.33, 1.33, 2.67, 1.67, 2.0, 0.67, 1.67, 1.0, 0.67, 0.67, 1.0, 1.0, 2.0, 1.67, 3.0, 3.67, 1.33, 1.67, 1.0, 2.0, 1.33, 2.33, 1.0, 2.0, 3.0, 1.67, 2.0, 1.0, 1.33, 1.33, 4.0, 1.0, 2.0, 1.33, 2.67, 0.33, 1.0, 1.67, 1.0, 1.67, 1.33, 2.0, 2.0]

NrTeams = [18,36]
Prizes = {18: [
                [[0,1],[2,15],[16,17]],
                [[0,5],[6,11],[12,17]],
                [[0,7],[8,9],[10,17]],
                [
                    [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], 
                    [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], 
                    [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17]
                ]
              ],
          36: [
                [[0,3],[4,31],[32,35]],
                [[0,11],[12,23],[24,35]],
                [[0,15],[16,17],[18,35]],
                [
                    [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], 
                    [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], 
                    [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], 
                    [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23], 
                    [24, 24], [25, 25], [26, 26], [27, 27], [28, 28], [29, 29], 
                    [30, 30], [31, 31], [32, 32], [33, 33], [34, 34], [35, 35]
                ]
             ]
}

class Team:
    def __init__(self, index, home_strength, away_strength):
        self.index = index
        self.home_strength = home_strength
        self.away_strength = away_strength

        self.points = 0
        self.GF = 0 # goals for
        self.GA = 0 # goals against


def WeightEdgeEP(i,h_team,a_team):
    w = 0
    p_i = i.points
    # Weight of the edge {j,k} when considering the elimination problem for i
    if i.index == h_team.index:
        if a_team.points >= p_i+4:
            w = 1
    elif i.index == a_team.index:
        if h_team.points >= p_i+4:
            w = 1
    else:
        # Define p_min and p_max for the edge e = {h_team, a_team}
        p_e_min = min(h_team.points, a_team.points)
        p_e_max = max(h_team.points, a_team.points)

        if p_e_min >= p_i + 4:
            w = 2
        elif p_e_min == p_i+3:
            w = 1
        elif p_e_max >= p_i + 4:
            w = 1
        elif p_e_max == p_i+3 and p_e_min >= p_i+1:
            w = 1
        else:
            w = 0
    return w

def WeightEdgeGPP(i, h_team, a_team):
    w = 0
    p_i = i.points
    
    # CASE: i is in the edge e (i in e)
    # The logic applies to the opponent j
    if i.index == h_team.index:
        if a_team.points >= p_i-3:
            w = 1
    elif i.index == a_team.index:
        if h_team.points >= p_i-3:
            w = 1
    else:
        p_e_min = min(h_team.points, a_team.points)
        p_e_max = max(h_team.points, a_team.points)
        
        if p_e_min >= p_i - 1:
            w = 2
        elif p_e_max >= p_i:
            if p_e_min >= p_i-3:
                w = 2 
            else:
                w = 1
        elif p_e_max >= p_i-3:
            w = 1
        else:
            w = 0
            
    return w

class IP:
    def __init__(self, Teams, M_bool, Prizes): 
        # M = set of possible matches
        self.Teams = Teams
        self.Prizes = Prizes
        # M_bool[i][j] = True if match (i,j) is possible and False if not
        self.M = []
        for i in self.Teams:
            for j in self.Teams:
                if i.index > j.index and M_bool[i.index][j.index]:
                    self.M.append([i,j])
        self.model = gp.Model("MTFPP")
        self.model.setParam("OutputFlag", 0)
        self.x = {}
        # y_ep[i, k]: Elimination indicator for team i at rank k
        self.y_ep = {}
        # y_gpp[i, k]: Guaranteed placement indicator for team i at rank k
        self.y_gpp = {}
        # z[i, j]: Band indicator for team i in band B_j
        self.z = self.model.addVars(len(Teams), len(Prizes), vtype=GRB.BINARY, name="z")

        bigM = 1000

        for m, match in enumerate(self.M):
            # x[m] = 1 if match m is selected or not
            self.x[m] = self.model.addVar(vtype=GRB.BINARY)
        for i in self.Teams:
            c0 = gp.LinExpr(0)
            for m, match in enumerate(self.M):
                if i.index == match[0].index or i.index == match[1].index:
                    c0 += self.x[m]
            self.model.addConstr(c0 == 1)

        for i in self.Teams:
            for p, [b_start, b_end] in enumerate(self.Prizes):
                if p < len(self.Prizes)-1:
                    c1 = gp.LinExpr(0)
                    c2 = gp.LinExpr(0)
                    k = b_end+1 # b_end is the position in the interval
                    self.y_ep[i.index, k] = self.model.addVar(vtype=GRB.BINARY)
                    self.y_gpp[i.index, k] = self.model.addVar(vtype=GRB.BINARY)
                    for m, match in enumerate(self.M):
                        c1 += WeightEdgeEP(i, match[0], match[1])*self.x[m]
                        c2 += WeightEdgeGPP(i, match[0], match[1])*self.x[m]
                    self.model.addConstr(bigM*self.y_ep[i.index, k] >= c1 - k + 1)
                    self.model.addConstr(bigM*self.y_gpp[i.index, k] >= k - c2)

        for i in self.Teams:
            # if a team is guaranteed to finish in the first b_end+1 positions: done
            self.model.addConstr(self.z[i.index, 0] >= self.y_gpp[i.index, self.Prizes[0][1]+1])
            # if a team is eliminated from the position b_end+1 in the second last interval: 
            # it is sure to finish in the last interval so also done!
            self.model.addConstr(self.z[i.index, len(self.Prizes)-1] >= self.y_ep[i.index, self.Prizes[len(self.Prizes)-1][0]])
            for p in range(1, len(self.Prizes)-1):
                b_start = self.Prizes[p][0]
                b_end = self.Prizes[p][1]+1
                self.model.addConstr(self.z[i.index, p] >= self.y_ep[i.index, b_start] + self.y_gpp[i.index, b_end] - 1)

        self.obj = gp.LinExpr(0)
        for i in self.Teams:
            for p in range(len(self.Prizes)):
                self.obj += self.z[i.index, p]
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def solve(self):
        self.model.optimize()

        if self.model.Status != 2:
            print(f"Could not find optimal solution!!")
            return -1
        else:
            return self.model.ObjVal

    def print_solution(self):
        for m, match in enumerate(self.M):
            if self.x[m].X > 0.9:
                print(f'Select match {match[0].index},{match[1].index}')
        for p in range(len(self.Prizes)):
            for i in self.Teams:
                if self.z[i.index, p].X > 0.1:
                    print(f"Team {i.index} has prize {p} fixed!")
        for i in self.Teams:
            for p, [b_start, b_end] in enumerate(self.Prizes):
                if p < len(self.Prizes)-1:
                    k = b_end+1 # b_end is the position in the interval
                    if self.y_ep[i.index, k].X > 0.1:
                        print(f'{i.index} is eliminated for the top {k}')
                    if self.y_gpp[i.index, k].X > 0.1:
                        print(f'{i.index} is guaranteed to finish in the top {k}')

    def ReturnLastRound(self):
        round_ = []
        for m, match in enumerate(self.M):
            if self.x[m].X > 0.9:
                round_.append([match[0], match[1]])
        return [round_]

def get_sampled_distributions(n,d):

    L_min=0.2281
    L_max=3.9847 # based on Fitting.py in CL code, see function
    # HistoricalLambdas, average min and max home goals scored in the UEFA CL,
    # for seasons 2005-2006 till 2024-2025

    # Calculate the range width
    L_range = L_max - L_min
    
    # 1. Uniform (alpha=1, beta=1)
    if d == 0:
        return L_min + L_range * np.random.beta(1, 1, n)
    
    # 2. Left Tail (Skewed Right: alpha=1, beta=4)
    # Most values near L_min, thinning out towards L_max
    if d == 1:
        return L_min + L_range * np.random.beta(1, 4, n)
    
    # 3. Right Tail (Skewed Left: alpha=4, beta=1)
    # Most values near L_max, thinning out towards L_min
    if d == 2:
        return L_min + L_range * np.random.beta(4, 1, n)
    
    # 4. Strong Center (alpha=10, beta=10)
    # Highly concentrated around the midpoint (L_min + L_max) / 2
    if d == 3:
        return L_min + L_range * np.random.beta(10, 10, n)

    if d == 4:
        return np.random.choice(real_lambda_h_CL, size=n, replace=True)

    if d == 5:
        return [1]*n


def Circle(teams,r):
    schedule = []
    n2 = int(n/2)

    for s in range(r):
        round_ = []
        for i in range(1, n2):
            if i % 2 == 0:
                match = [teams[i], teams[n-i-1]]
            else:
                match = [teams[n-i-1], teams[i]]
            round_.append(match)
        if r % 2 == 0:
            match = [teams[0], teams[n-1]]
        else:
            match = [teams[n-1], teams[0]]
        round_.append(match)
        teams.insert(1, teams.pop())  # Rotate the elements
        schedule.append(round_)
    return schedule


def VizingSchedules(n, r):
    schedules = []
    # Number of edges
    folder = "Input"
    file_name = "Vizing_n" + str(n) + "_r" + str(r) + ".txt"
    file_path = os.path.join(folder, file_name)
    prev_s = -1
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if int(row[0]) != prev_s:
                if prev_s > -1:
                    schedules.append(schedule)
                prev_s = int(row[0])
                schedule = [[] for s in range(r)]
            schedule[int(row[1])].append([int(row[2]), int(row[3])])
    return schedules


def test_schedule_feasible(Teams, schedule):
    success = True
    nr_match_present = {i.index: {j.index: 0 for j in Teams} for i in Teams}
    for r, round_ in enumerate(schedule):
        nr_team_present_in_round = {i.index: 0 for i in Teams}
        for match in round_:
            nr_team_present_in_round[match[0].index] += 1
            nr_team_present_in_round[match[1].index] += 1
            nr_match_present[match[0].index][match[1].index] += 1
            nr_match_present[match[1].index][match[0].index] += 1
        for tm in Teams:
            if nr_team_present_in_round[tm.index] != 1:
                print(f"Team {tm.index} is present {nr_team_present_in_round[tm.index]} times in round {r}")
                success = False
    for i in Teams:
        for j in Teams:
            if nr_match_present[i.index][j.index] > 1:
                print(f"The match {i.index}-{j.index} is present {nr_match_present[i.index][j.index]} times!")
                success = False
    if not success:
        # print schedule:
        for r, round_ in enumerate(schedule):
            print(f"---------")
            print(f"Round {r}")
            print(f"---------")
            for match in round_:
                print(f"{match[0].index} vs {match[1].index}")
        sys.exit()


def sample_match_outcome(h_team, a_team, rng):
    lambda_home = h_team.home_strength # no home advantage
    lambda_away = a_team.away_strength 

    HG = rng.poisson(lambda_home)
    AG = rng.poisson(lambda_away)

    if HG > AG:
        h_team.points += 3
    elif HG < AG:
        a_team.points += 3
    else:
        h_team.points += 1
        a_team.points += 1


def NrTeamsPrizeFixedEveryPositionIsPrize(Teams):
    nr = 0
    n = len(Teams)-1
    for i, tm in enumerate(Teams):
        if i == 0:
            if tm.points > Teams[1].points+3:
                nr += 1
        elif i == n:
            if tm.points+3 < Teams[n-1].points:
                nr += 1
        else:
            if tm.points > Teams[i+1].points+3 and tm.points+3 < Teams[i-1].points:
                nr += 1
    return nr


def NrTeamsPrizeFixed(round_, Prizes):
    nr = 0
    for i in Teams:
        total_w_ep = 0
        total_w_gpp = 0
        for match in round_:
            total_w_ep += WeightEdgeEP(i,match[0],match[1])
            # print(f"{match[0].index}-{match[1].index} gets score {WeightEdgeEP(i,match[0],match[1])}")
            total_w_gpp += WeightEdgeGPP(i,match[0],match[1])
        # check for which prize a team is eliminated (stop as soon as we find a prize it is not eliminated for)
        if total_w_gpp < Prizes[0][1]+1: # first check: guaranteed to finish in the first interval?
            nr += 1
            # print(f"Team {i.index} is guaranteed to finish in the top {Prizes[0][1]+1}")
        elif total_w_ep >= Prizes[-1][0]: # second check: eliminated from second last interval?
            nr += 1
            # print(f"Team {i.index} is eliminated from the top {Prizes[-1][0]}")
        else:
            for p in range(1, len(Prizes)-1):
                if total_w_gpp < Prizes[p][1]+1 and total_w_ep >= Prizes[p][0]:
                    # print(f"Team {i.index} is eliminated from the top {Prizes[p][0]} and guaranteed to finish in the top {Prizes[p][1]+1}")
                    nr += 1
                    break
    return nr


if __name__ == '__main__':
    n = int(sys.argv[1]) # nr of teams
    r = int(sys.argv[2]) # nr of rounds
    b = int(sys.argv[3]) # prizes
    sd = int(sys.argv[4]) # distribution from which we sample strength values
    seed = int(sys.argv[5]) # seed
    nr_simulations = int(sys.argv[6]) # number of simulations
    print(f"Test configuration with {n} teams, {r} rounds, set of prizes {b} and strength distribution {sd}, with seed {seed} and {nr_simulations} simulations")
    rng = np.random.default_rng(seed)

    B = Prizes[n][b]

    data = {"diff": [], "static": [], "dynamic": []}

    schedules = VizingSchedules(n, r) # we only have 100k vizing schedules!!

    for simul in range(nr_simulations):
        if simul > 0 and simul % 100 == 0:
            print(simul)
        # sample n strength values from the distribution d
        samples = get_sampled_distributions(n,sd)
        # construct the teams
        Teams = []
        for i in range(n):
            l_h = samples[i]
            l_a = 0.9111*l_h # also based on CL data
            tm = Team(i, l_h, l_a)
            Teams.append(tm)

        # All possible matches:
        M = [[True] * n for _ in range(n)]

        # construct a random schedule with vizing or circle
        schedule = [[] for s in range(r)]
        for s in range(len(schedules[simul])):
            for match in schedules[simul][s]:
                schedule[s].append([Teams[match[0]], Teams[match[1]]])
        # schedule = Circle(Teams, r)
        test_schedule_feasible(Teams, schedule)
        # play all matches until the final round 
        for s in range(r-1):
            for match in schedule[s]:
                sample_match_outcome(match[0], match[1], rng)
                M[match[0].index][match[1].index] = False
                M[match[1].index][match[0].index] = False

        # rank the teams
        '''
        Teams.sort(key=lambda team: team.points, reverse=True)
        for tm in Teams:
            print(f'{tm.index}: {tm.points}')
        '''

        # count number of teams whose prizes are fixed if we would play the last round as is

        # print(f"Final round matches:")
        '''
        for match in schedule[-1]:
            print(f"{match[0].index}, {match[1].index}")
        '''
        if b < 3:
            nr_static = NrTeamsPrizeFixed(schedule[-1], B)
        else:
            Teams.sort(key=lambda team: team.points, reverse=True)
            nr_static = NrTeamsPrizeFixedEveryPositionIsPrize(Teams)
        # print(f"Nr teams whose prizes are fixed = {nr_static}")
        # now, construct a new schedule with the goal to minimize the number of teams with fixed prizes
        ip = IP(Teams, M, B)
        nr_dynamic = ip.solve()
        # print(f"{nr_dynamic} teams their prizes are fixed according to IP!!")
        # ip.print_solution()
        schedule = ip.ReturnLastRound()
        
        if b < 3:
            nr_dynamic2 = NrTeamsPrizeFixed(schedule[-1], B)
        else:
            nr_dynamic2 = NrTeamsPrizeFixedEveryPositionIsPrize(Teams)
        if nr_dynamic != nr_dynamic2:
            print(f"{nr_dynamic} != {nr_dynamic2}")
            Teams.sort(key=lambda team: team.points, reverse=True)
            for tm in Teams:
                print(f'{tm.index}: {tm.points}')
            print(f"IP:")
            ip.print_solution()
            sys.exit()
        # print(f"Nr teams whose prizes are fixed = {nr_dynamic2}")
        data["diff"].append(nr_static-nr_dynamic)
        data["static"].append(nr_static)
        data["dynamic"].append(nr_dynamic)
    minimum = min(data["diff"])
    maximum = max(data["diff"])
    mean = sum(data["diff"]) / len(data["diff"])
    # print(f"Min: {minimum}, Max: {maximum}, Mean: {mean}")

    u_stat, p_val = stats.mannwhitneyu(data["static"], data["dynamic"])
    # Is this data not related and should we not do a wilcoxon signed-rank test instead?

    # Nice boxplots:

    data_boxplot = pd.DataFrame({
        'Setting': ['static'] * len(data['static']) + ['dynamic'] * len(data['dynamic']),
        'Rounds': data['static'] + data['dynamic']
    })

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.set_theme(style="ticks")

    dark_green = '#1E8449'
    dark_purple = '#BA4A00'
    pal = {'static': dark_green, 'dynamic': dark_purple}

    light_green = '#A9DFBF'
    light_purple = '#EDBB99'
    face_pal = {'static': light_green, 'dynamic': light_purple}

    hue_order = ['static', 'dynamic']

    boxprops = {'edgecolor': 'k', 'linewidth': 2}
    lineprops = {'color': 'k', 'linewidth': 2}

    boxplot_kwargs = {
        'boxprops': boxprops, 
        'medianprops': lineprops,
        'whiskerprops': lineprops, 
        'capprops': lineprops,
        'width': 0.9, 
        'gap': 0.0,
        'palette': face_pal,
        'hue_order': hue_order
    }

    stripplot_kwargs = {
        'edgecolor': 'k',
        'linewidth': 0.6, 
        'size': 4, 
        'alpha': 0.8,
        'palette': pal, 
        'hue_order': hue_order,
    }

    sns.boxplot(
        x='Setting', y='Rounds', hue='Setting', data=data_boxplot, 
        ax=ax, fliersize=0, **boxplot_kwargs
    )

    sns.stripplot(
        x='Setting', y='Rounds', hue='Setting', data=data_boxplot, 
        ax=ax, dodge=True, jitter=0.2, **stripplot_kwargs
    )

    if ax.legend_ is not None:
        ax.legend_.remove()

    plt.ylim(0, n)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    sns.despine()
    plt.tight_layout()

    Figure_name = os.path.join("Figures", "n" + str(n) + "_r" + str(r) + "_b" + str(b) + "_sd" + str(sd))
    plt.savefig(Figure_name)

    folder = "Results"
    file_name = "n" + str(n) + "_r" + str(r) + "_b" + str(b) + "_sd" + str(sd) + '.txt'
    print(f'Save file as {file_name}')

    file_path = os.path.join(folder, file_name)
    with open(file_path, "w") as file:
        file.write(f"{minimum},{maximum},{mean},")
        file.write(f"{round(u_stat,1)},{round(p_val,4)}")

    # try to read the three values!
