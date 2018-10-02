# coding: utf-8
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



class FindPosition(object):
	def __init__(self, file = 'CompleteDataset.csv'):
		self.df = pd.read_csv(file, dtype=str)
		self.process_csv()
		self.fit_model()
		self.define_testcase()
		self.generate_knn_clf()
		self.console_interaction()

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
		self.df = self.df[columns_needed_rearranged]

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
		self.mapping_pos_int = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CF': 7,
								'CDM': 8, 'CB': 9,'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}
		self.mapping_int_pos = ["ST","RW","LW","RM","CM","LM","CAM","CF","CDM","CB","LB","RB","RWB","LWB"]

		# mapping positions from string to int
		self.df_new_normalized_all = self.df_new_normalized_all.replace({'Preferred Positions': self.mapping_pos_int})
		# self.df_new_normalized_all.iloc[::1000, ]
	
	def fit_model(self):
		# Fit the model:
		self.X_train_all, self.X_test_all, self.y_train_all, self.y_test_all = train_test_split(\
			self.df_new_normalized_all.iloc[:, :-1], self.df_new_normalized_all.iloc[:, -1], test_size=0.05, random_state=0)

	def generate_knn_clf(self):
		self.clf_knn = KNeighborsClassifier(n_neighbors=150)
		self.clf_knn.fit(self.X_train_all, self.y_train_all)
		print("The classification accuracy is: "),
		print(self.clf_knn.score(self.X_test_all, self.y_test_all))
		print("This accuracy is affected by L/R positions, since players "
			  "in each sides may have similar attributes.")

	def show_df(self, df):
		df.head()
		df.columns
		df.iloc[::500, :]
		print('X train shape: {}'.format(self.X_train_all.shape))
		print('X test shape: {}'.format(self.X_test_all.shape))
		print('y train shape: {}'.format(self.y_train_all.shape))
		print('y test shape: {}'.format(self.y_test_all.shape))

	def rearrange_column(self, df):
		"""rearrrange column so that testing data is listed in the same order with training data"""
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

		# This shall be a Center Back，who is excel in tackle and is strong physically.
		self.test_case = pd.DataFrame({'Aggression': [20], 'Crossing': [20], 'Curve': [20], 'Dribbling': [20],
								  'Finishing': [20], 'Free kick accuracy': [20], 'Heading accuracy': [88],
								  'Long shots': [20], 'Penalties': [20], 'Shot power': [70], 'Volleys': [88],
								  'Short passing': [20], 'Long passing': [20], 'Interceptions': [29],
								  'Marking': [99], 'Sliding tackle': [99], 'Standing tackle': [99],
								  'Strength': [95], 'Vision': [85], 'Acceleration': [80], 'Agility': [75],
								  'Reactions': [80], 'Stamina': [85], 'Balance': [90], 'Ball control': [70],
								  'Composure': [70], 'Jumping': [95], 'Sprint speed': [91], 'Positioning': [95]},
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

		# The output of predict_proba is the probability of
		# ST、RW、LW、RM、CM、LM、CAM、CF、CDM、CB、LB、RB、RWB、LWB respectively,
		# in which W is Wing Forward，ST/CF is Striker/Forward，M id Midfielder，
		# B is Back，and L/R stands for Left and Right Side
		print("\nThe best position predicted for this player is:"),
		print(self.mapping_int_pos[self.clf_knn.predict(self.test_case)[0]])
		print(self.clf_knn.predict_proba(self.test_case))


fp = FindPosition()