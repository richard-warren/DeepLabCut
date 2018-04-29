:: analyze videos from multiple sessions using DeepLabCut
:: must be run from root directory of DeepLabCut
:: takes top and bot views in folder for each session, concatenates them, and moves them to Evaluation-Tools\Videos forlder in DeepLabCut directory (which must be set as home directory)
:: then runs DeepLabCut analysis, which generates a spreadsheet that is then moved back to the sessions folder


:: settings
set sessions=180122_001 180122_002 180122_003
set sessionFolder=C:\Users\rick\Google Drive\columbia\obstacleData\sessions\
set bitRate=10M
set frameRate=250




:: prevent all commands from displaying in command prompt
echo off

:: concatonate top and bot videos
for %%s in (%sessions%) do (

	echo ---------- ANALYZING SESSION: %%s ----------
	
	:: concatenate top and bot vids and place in vidFolder
	echo concatenating top and bottom views...
	ffmpeg -y -loglevel panic -stats -i "%sessionFolder%%%s\runTop.mp4" -i "%sessionFolder%%%s\runBot.mp4" ^
		-filter_complex 'vstack' -vcodec mpeg4 -vb %bitRate% Evaluation-Tools\Videos\runTopBot.avi

	:: analyze with DeepLabCut
	echo analyzing with DeepLabCut
	cd Evaluation-Tools
	python -W ignore %evalDirDLC%AnalyzeVideos.py
	
	:: move results back to session directory and delete concatenated video
	move /y Videos\trackedFeaturesRaw.csv "%sessionFolder%%%s\trackedFeaturesRaw.csv"
	del Videos\runTopBot.avi
	cd ..

)

