:: analyze videos from multiple sessions using DeepLabCut
:: must be run from root directory of DeepLabCut
:: takes top and bot views in folder for each session, concatenates them, and moves them to Evaluation-Tools\Videos forlder in DeepLabCut directory (which must be set as home directory)
:: then runs DeepLabCut analysis, which generates a spreadsheet that is then moved back to the sessions folder

:: prevent all commands from displaying in command prompt
echo off


:: settings
:: set sessions=180608_005 180609_000 180609_001 180609_002 180609_003 180609_004 180609_005 180612_000 180612_001 180612_002 180612_003 180612_004
:: set sessions=180615_001 180615_002 180615_003 180615_004 180615_005
set sessions=180613_000 180613_001 180613_002 180613_003 180613_004 180613_005
set sessionFolder=C:\Users\rick\Google Drive\columbia\obstacleData\sessions\
set bitRate=10M
:: set frameRate=2.5






:: concatonate top and bot videos
for %%s in (%sessions%) do (

	echo ---------- ANALYZING SESSION: %%s ----------
	
	:: concatenate top and bot vids and place in vidFolder
	echo concatenating top and bottom views...
	::ffmpeg -y -loglevel panic -stats -r 250 -i "%sessionFolder%%%s\runTop.mp4" -i "%sessionFolder%%%s\runBot.mp4" ^
	::	-filter_complex 'vstack' -vcodec mpeg4 -vb %bitRate% Evaluation-Tools\Videos\runTopBot.avi
	ffmpeg -y -loglevel panic -stats -r 250 -i "%sessionFolder%%%s\runTop.mp4" -i "%sessionFolder%%%s\runBot.mp4" ^
		-filter_complex vstack,scale=198:203 -vcodec mpeg4 -vb %bitRate% Evaluation-Tools\Videos\runTopBot.avi

	:: analyze with DeepLabCut
	echo analyzing with DeepLabCut
	cd Evaluation-Tools
	python %evalDirDLC%AnalyzeVideos.py
	
	:: move results back to session directory and delete concatenated video
	move /y Videos\trackedFeaturesRaw.csv "%sessionFolder%%%s\trackedFeaturesRaw.csv"
	del Videos\runTopBot.avi
	cd ..

)

