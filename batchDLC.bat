:: analyze videos from multiple sessions using DeepLabCut
:: takes top and bot views in folder for each session, concatenates them, and moves them to Evaluation-Tools\Videos forlder in DeepLabCut directory (which must be set as home directory)
:: then runs DeepLabCut analysis, which generates a spreadsheet that is then moved back to the sessions folder

:: prevent all commands from displaying in command prompt
echo off


:: settings
set pythonPath=C:\Users\rick\Anaconda3\envs\deepLabCut\python.exe
set sessionFolder=Z:\loco\obstacleData\sessions\
set bitRate=10M




:: concatonate top and bot videos
for %%s in (%*) do (

	echo ---------- ANALYZING SESSION: %%s ----------

	:: move video into vidFolder
	copy "%sessionFolder%%%s\run.mp4" Evaluation-Tools\Videos\run.avi

	
	:: analyze with DeepLabCut
	echo analyzing with DeepLabCut
	cd Evaluation-Tools
	%pythonPath% AnalyzeVideos.py
	
	:: move results back to session directory and delete concatenated video
	move /y Videos\trackedFeaturesRaw.csv "%sessionFolder%%%s\trackedFeaturesRaw.csv"
	del Videos\run.avi
	cd ..
)

