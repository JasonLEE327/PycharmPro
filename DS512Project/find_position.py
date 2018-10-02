# coding: utf-8
import pandas as pd
import numpy as np
# import seaborn as sns
# sns.set_style("darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


class FindPosition(object):
	def __init__(self, file = 'CompleteDataset.csv', data = None, attributes = None):
		# import os
		# print os.getcwd()
		if type(data) == type(None):
			print "load data from csv"
			self.df_raw = pd.read_csv(file, dtype=str)
		else:
			self.df_raw = data
		# print self.df_raw
		self.process_csv()
		self.define_testcase()
		self.generate_knn_clf()
		if type(attributes) == type(None):
			print ("load attributes from python input.")
		else:
			dic = {}
			print "load attributes from flask. player's attributes are:"
			for i in range(len(attributes)):
				print attributes[i],
				dic[self.attributes[i]] = [attributes[i]]
			self.test_case = pd.DataFrame(data=dic)
			self.test_case = self.rearrange_column(self.test_case)
		# self.console_interaction()
		self.predict()

	def process_csv(self):
		# Gather only columns that we need for this analysis purpose:
		# attack attribute first, then defence, then mixed
		columns_needed_rearranged = ['Aggression', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
									 'Free kick accuracy', 'Heading accuracy', 'Long shots', 'Penalties', 'Shot power',
									 'Volleys',
									 'Short passing', 'Long passing',
									 'Interceptions', 'Marking', 'Sliding tackle', 'Standing tackle',
									 'Strength', 'Vision', 'Acceleration', 'Agility',
									 'Reactions', 'Stamina', 'Balance', 'Ball control', 'Composure', 'Jumping',
									 'Sprint speed', 'Positioning', 'Preferred Positions']
		self.df = self.df_raw[columns_needed_rearranged]

		# We don't want to classify GK because it will be too obvious:
		self.df['Preferred Positions'] = self.df['Preferred Positions'].str.strip()
		self.df = self.df[self.df['Preferred Positions'] != 'GK']
		self.df.head()

		# Check any missing data:
		if self.df.isnull().values.any() == True:
			print("Error in DataFrame")

		# All possible outcome for preferred position:
		position = self.df['Preferred Positions'].str.split().apply(lambda x: x[0]).unique()
		# p

		# Handle players with multiple preferred positions: duplicate a set of data for each
		# copy a structure
		self.df_new = self.df.copy()
		self.df_new.drop(self.df_new.index, inplace=True)
		for i in position:
			df_temp = self.df[self.df['Preferred Positions'].str.contains(i)]
			df_temp['Preferred Positions'] = i
			self.df_new = self.df_new.append(df_temp, ignore_index=True)

		# Some of the attributes have '+/-' sign, let's perform the calculation rather than keeping them as string:
		cols = [col for col in self.df_new.columns if col not in ['Preferred Positions']]

		for i in cols:
			self.df_new[i] = self.df_new[i].apply(lambda x: eval(x) if isinstance(x, str) else x)

		# self.df_new.iloc[::500, :]
		self.df_new_normalized_all = self.df_new.copy()
		# use old version
		# self.mapping_pos_int = {'CF': 0, 'ST': 1, 'RW': 2, 'LW': 3, 'RM': 4, 'CM': 5, 'LM': 6, 'CAM': 7,
		#                         'CDM': 8, 'CB': 9, 'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}
		# self.mapping_int_pos = ["CF", "ST", "RW", "LW", "RM", "CM", "LM", "CAM", "CDM", "CB", "LB", "RB", "RWB", "LWB"]

		self.mapping_pos_int = {'CF': 0, 'ST': 1, 'RW': 2, 'LW': 2, 'RM': 3, 'LM': 3, 'CAM': 4, 'CM': 5,
								'CDM': 6, 'CB': 7,'LB': 8, 'RB': 8, 'RWB': 9, 'LWB': 9}
		self.mapping_int_pos = ["CF","ST","WF","SM","CAM","CM","CDM","CB","SB","WB"]

		# mapping positions from string to int
		self.df_new_normalized_all = self.df_new_normalized_all.replace({'Preferred Positions': self.mapping_pos_int})

		# look how many players are in the same position
		# print self.df_new_normalized_all.iloc[::200, ]
		# re-weighting
		df_temp = self.df_new_normalized_all[self.df_new_normalized_all['Preferred Positions'] == 0]
		self.df_new_normalized_all = self.df_new_normalized_all.append(df_temp, ignore_index=True)
		self.df_new_normalized_all = self.df_new_normalized_all.append(df_temp, ignore_index=True)
		self.df_new_normalized_all = self.df_new_normalized_all.append(df_temp, ignore_index=True)
		df_temp = self.df_new_normalized_all[self.df_new_normalized_all['Preferred Positions'] == 2]
		self.df_new_normalized_all = self.df_new_normalized_all.append(df_temp, ignore_index=True)
		self.df_new_normalized_all = self.df_new_normalized_all.append(df_temp, ignore_index=True)
		# df_temp = self.df_new_normalized_all[self.df_new_normalized_all['Preferred Positions'] == 3]
		# self.df_new_normalized_all = self.df_new_normalized_all.append(df_temp, ignore_index=True)
		# self.df_new_normalized_all = self.df_new_normalized_all.append(df_temp, ignore_index=True)

		# for i in range(len(self.mapping_int_pos)):
		# 	print self.mapping_int_pos[i]+'=',
		# 	print len(self.df_new_normalized_all[self.df_new_normalized_all['Preferred Positions'] == i].iloc[::, :-1])


		# split dataset
		self.X_train_all, self.X_test_all, self.y_train_all, self.y_test_all = train_test_split(\
			self.df_new_normalized_all.iloc[:, :-1], self.df_new_normalized_all.iloc[:, -1], test_size=0.05, random_state=0)

	def generate_knn_clf(self):
		self.clf_knn = KNeighborsClassifier(n_neighbors=50)
		self.clf_knn.fit(self.X_train_all, self.y_train_all)
		# print("The classification accuracy is: "),
		# print(self.clf_knn.score(self.X_test_all, self.y_test_all))


		# self.clf_knn = AdaBoostClassifier(n_estimators=10)
		# self.clf_knn.fit(self.X_train_all, self.y_train_all)
		# print("The classification accuracy is: "),
		# print(self.clf_knn.score(self.X_test_all, self.y_test_all))

	def show_df(self, df):
		df.head()
		df.columns
		df.iloc[::500, :]
		print('X train shape: {}'.format(self.X_train_all.shape))
		print('X test shape: {}'.format(self.X_test_all.shape))
		print('y train shape: {}'.format(self.y_train_all.shape))
		print('y test shape: {}'.format(self.y_test_all.shape))

	def rearrange_column(self, df):
		"""rearrange column so that testing data is listed in the same order with training data"""
		self.attributes = ['Aggression', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
									 'Free kick accuracy', 'Heading accuracy', 'Long shots', 'Penalties', 'Shot power',
									 'Volleys',
									 'Short passing', 'Long passing',
									 'Interceptions', 'Marking', 'Sliding tackle', 'Standing tackle',
									 'Strength', 'Vision', 'Acceleration', 'Agility',
									 'Reactions', 'Stamina', 'Balance', 'Ball control', 'Composure', 'Jumping',
									 'Sprint speed', 'Positioning']
		df = df[self.attributes]
		return df

	def define_testcase(self):
		# This is C. Ronaldo
		# self.test_case = pd.DataFrame({'Aggression':[63],'Crossing':[85], 'Curve':[81], 'Dribbling':[91],
		# 							 'Finishing':[94],'Free kick accuracy':[76], 'Heading accuracy':[88],
		# 							 'Long shots':[92],'Penalties':[85], 'Shot power':[94], 'Volleys':[88],
		# 							 'Short passing':[83], 'Long passing':[77],'Interceptions':[29],
		# 							 'Marking':[22], 'Sliding tackle':[23], 'Standing tackle':[31],
		# 							 'Strength':[80], 'Vision':[85], 'Acceleration':[89], 'Agility':[89],
		# 							 'Reactions':[96], 'Stamina':[92], 'Balance':[63], 'Ball control':[93],
		# 							 'Composure':[95],'Jumping':[95],'Sprint speed':[91], 'Positioning':[95]},
		# 							 dtype=int)

		# This is L. Messi WF
		# self.test_case = pd.DataFrame({'Aggression':[48],'Crossing':[85], 'Curve':[89], 'Dribbling':[97],
		# 							 'Finishing':[95],'Free kick accuracy':[90], 'Heading accuracy':[71],
		# 							 'Long shots':[88],'Penalties':[74], 'Shot power':[85], 'Volleys':[85],
		# 							 'Short passing':[88], 'Long passing':[87],'Interceptions':[22],
		# 							 'Marking':[13], 'Sliding tackle':[26], 'Standing tackle':[28],
		# 							 'Strength':[59], 'Vision':[90], 'Acceleration':[92], 'Agility':[90],
		# 							 'Reactions':[95], 'Stamina':[73], 'Balance':[95], 'Ball control':[95],
		# 							 'Composure':[96],'Jumping':[68],'Sprint speed':[87], 'Positioning':[93]},
		# 							 dtype=int)

		# This is Neymar WF
		# self.test_case = pd.DataFrame({'Aggression':[56],'Crossing':[75], 'Curve':[81], 'Dribbling':[96],
		# 							 'Finishing':[89],'Free kick accuracy':[84], 'Heading accuracy':[62],
		# 							 'Long shots':[77],'Penalties':[81], 'Shot power':[80], 'Volleys':[83],
		# 							 'Short passing':[81], 'Long passing':[75],'Interceptions':[36],
		# 							 'Marking':[21], 'Sliding tackle':[33], 'Standing tackle':[24],
		# 							 'Strength':[53], 'Vision':[80], 'Acceleration':[94], 'Agility':[96],
		# 							 'Reactions':[88], 'Stamina':[78], 'Balance':[82], 'Ball control':[95],
		# 							 'Composure':[92],'Jumping':[61],'Sprint speed':[90], 'Positioning':[90]},
		# 							 dtype=int)

		# This is Iniesta SM
		# self.test_case = pd.DataFrame({'Aggression':[58],'Crossing':[77], 'Curve':[80], 'Dribbling':[90],
		# 							 'Finishing':[70],'Free kick accuracy':[70], 'Heading accuracy':[54],
		# 							 'Long shots':[71],'Penalties':[71], 'Shot power':[65], 'Volleys':[74],
		# 							 'Short passing':[92], 'Long passing':[86],'Interceptions':[66],
		# 							 'Marking':[57], 'Sliding tackle':[56], 'Standing tackle':[57],
		# 							 'Strength':[58], 'Vision':[94], 'Acceleration':[72], 'Agility':[79],
		# 							 'Reactions':[88], 'Stamina':[58], 'Balance':[84], 'Ball control':[94],
		# 							 'Composure':[89],'Jumping':[52],'Sprint speed':[71], 'Positioning':[84]},
		# 							 dtype=int)

		# This is Sergio Ramos
		self.test_case = pd.DataFrame({'Aggression': [84], 'Crossing': [66], 'Curve': [73], 'Dribbling': [61],
								  'Finishing': [60], 'Free kick accuracy': [67], 'Heading accuracy': [91],
								  'Long shots': [55], 'Penalties': [68], 'Shot power': [78], 'Volleys': [66],
								  'Short passing': [78], 'Long passing': [72], 'Interceptions': [88],
								  'Marking': [86], 'Sliding tackle': [91], 'Standing tackle': [89],
								  'Strength': [81], 'Vision': [63], 'Acceleration': [75], 'Agility': [79],
								  'Reactions': [85], 'Stamina': [84], 'Balance': [60], 'Ball control': [84],
								  'Composure': [80], 'Jumping': [93], 'Sprint speed': [77], 'Positioning': [52]},
								  dtype=int)
		self.test_case = self.rearrange_column(self.test_case)

	def console_interaction(self):
		yn = raw_input("Do you want to input attributes manually (y/n):")
		if yn == 'y':
			dic = {}
			for i in range(29):
				value = input("Please input " + self.attributes[i] + ":")
				dic[self.attributes[i]] = [value]
			self.test_case = pd.DataFrame(data = dic)
			# print self.test_case

	def predict(self):
		# The output of predict_proba is the probability of
		# ST、RW、LW、RM、CM、LM、CAM、CF、CDM、CB、LB、RB、RWB、LWB respectively,
		# in which W is Wing Forward，ST/CF is Striker/Forward，M id Midfielder，
		# B is Back，and L/R stands for Left and Right Side

		best_pos = self.mapping_int_pos[self.clf_knn.predict(self.test_case)[0]]
		prob = self.clf_knn.predict_proba(self.test_case)
		print("\nThe best position predicted for this player is:"),
		print(best_pos)
		print ("Probability for each position is:")
		[prob2] = prob
		for i in range(len(self.mapping_int_pos)):
			print self.mapping_int_pos[i] + ' = ',
			print(prob2[i])

		return best_pos, prob


if __name__ == "__main__":
	fp = FindPosition()
	fp.console_interaction()
	fp.predict()