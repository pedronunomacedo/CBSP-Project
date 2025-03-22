import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import { message } from "antd";
import UploadBtn from "./buttons/UploadBtn";
import { AnimatePresence, motion } from "framer-motion";

const MusicInfo = ({ musicPlaying, setMusicPlaying, audioRef, musicName, setMusicName, handleMusicEnd }) => {
	const [displayedData, setDisplayedData] = useState([]);
	const [visibleWindowStart, setVisibleWindowStart] = useState(null);
	const [dataReady, setDataReady] = useState(false);

	const [bpmData, setBpmData] = useState([]);
	const [audioBPM, setAudioBPM] = useState(0);
	const [audioTempo, setAudioTempo] = useState("");
	const [songName, setSongName] = useState("");
	const [songArtist, setSongArtist] = useState("");

	const [mineBpmData, setMineBpmData] = useState([]);
	const [mineAudioBPM, setMineAudioBPM] = useState(0);
	const [mineAudioTempo, setMineAudioTempo] = useState("");
	const [mineDisplayedData, setMineDisplayedData] = useState([]);

	const [hybridBpmData, setHybridBpmData] = useState([]);
	const [hybridAudioBPM, setHybridAudioBPM] = useState(0);
	const [hybridAudioTempo, setHybridAudioTempo] = useState("");
	const [hybridDisplayedData, setHybridDisplayedData] = useState([]);

	const windowSize = 7; // Number of points to display in the visible window

	useEffect(() => {
		if (
			bpmData.length > 0 &&
			mineBpmData.length > 0 &&
			hybridBpmData.length > 0
		) {
			setDataReady(true);
		}
	}, [bpmData, mineBpmData, hybridBpmData]);

	// Handle file selection
	const handleFileChange = (file) => {
		setMusicName(file.name);
		console.log("Selected file:", file.name);
	};

	// Handle file upload and processing
	const handleUpload = async (file) => {
		if (!file) {
			message.error("Please select a file first.");
			return;
		}

		const formData = new FormData();
		formData.append("file", file);

		// Add the new audio data
		let newAudioData = {
			name: '',
			singer: '', 
			librosaBpm: '',
			librosaTempo: '',
			mineBmp: '',
			mineTempo: '',
			hybridBpm: '',
			hybridTempo: '',
		};

		try {
			const response = await fetch(
				"http://localhost:8000/api/v1/bpm_per_second",
				{
					method: "POST",
					body: formData,
				}
			);

			if (!response.ok) {
				throw new Error("Failed to upload file");
			}

			const data = await response.json();
			setBpmData(data.bpm_per_second);
			setAudioBPM(data.song_bpm);
			setAudioTempo(data.song_tempo);
			setSongName(data.song_name);
			setSongArtist(data.song_artist);

			newAudioData.name = data.song_name || file.name;
			newAudioData.singer = data.song_artist || '';
			newAudioData.librosaBpm = data.song_bpm;
			newAudioData.librosaTempo = data.song_tempo;

			// Initialize displayedData with the first windowSize points
			setDisplayedData(data.bpm_per_second.slice(0, windowSize));

			// Set the audio source to play immediately after upload
			if (audioRef.current) {
				audioRef.current.src = URL.createObjectURL(file);
			}
		} catch (error) {
			console.error("Error uploading file:", error);
			message.error("Failed to upload file. Please try again.");
		}

		// Get the BPM and tempo from my algorithm 
		try {
			const response = await fetch(
				"http://localhost:8000/api/v1/mine_bpm_per_second",
				{
					method: "POST",
					body: formData,
				}
			);

			if (!response.ok) {
				throw new Error("Failed to upload file");
			}

			const data = await response.json();
			setMineBpmData(data.bpm_per_second);
			setMineAudioBPM(data.song_bpm);
			setMineAudioTempo(data.song_tempo);

			newAudioData.mineBmp = data.song_bpm;
			newAudioData.mineTempo = data.song_tempo;

			// Initialize displayedData with the first windowSize points
			setMineDisplayedData(data.bpm_per_second.slice(0, windowSize));

			// Set the audio source to play immediately after upload
			if (audioRef.current) {
				audioRef.current.src = URL.createObjectURL(file);
			}
		} catch (error) {
			console.error("Error uploading file:", error);
			message.error("Failed to upload file. Please try again.");
		}

		// Get the BPM and tempo from my algorithm 
		try {
			const response = await fetch(
				"http://localhost:8000/api/v1/hybrid_bpm_per_second",
				{
					method: "POST",
					body: formData,
				}
			);

			if (!response.ok) {
				throw new Error("Failed to upload file");
			}

			const data = await response.json();
			setHybridBpmData(data.bpm_per_second);
			setHybridAudioBPM(data.song_bpm);
			setHybridAudioTempo(data.song_tempo);

			newAudioData.hybridBpm = data.song_bpm;
			newAudioData.hybridTempo = data.song_tempo;

			// Initialize displayedData with the first windowSize points
			setHybridDisplayedData(data.bpm_per_second.slice(0, windowSize));

			// Set the audio source to play immediately after upload
			if (audioRef.current) {
				audioRef.current.src = URL.createObjectURL(file);
			}
		} catch (error) {
			console.error("Error uploading file:", error);
			message.error("Failed to upload file. Please try again.");
		}

		// Get the existing audio data from local storage (if any)
		const storedAudioData = JSON.parse(localStorage.getItem("audio_data")) || [];
	
		storedAudioData.push(newAudioData);
	
		// Store the updated audio data back to local storage
		localStorage.setItem("audio_data", JSON.stringify(storedAudioData));


		if (
			bpmData.length > 0 &&
			mineBpmData.length > 0 &&
			hybridBpmData.length > 0
		) {
			setDataReady(true);
		}




		// DELETE:   Get the music peaks!
		// try {
		// 	const response = await fetch(
		// 		"http://localhost:8000/api/v1/detect_music",
		// 		{
		// 			method: "POST",
		// 			body: formData,
		// 		}
		// 	);

		// 	if (!response.ok) {
		// 		throw new Error("Failed to upload file");
		// 	}

		// 	const data = await response.json();
		// 	setMusicPeaks(data.peaks);

		// 	console.log("Peaks: ", data.peaks);
		// } catch (error) {
		// 	console.error("Error getting file frequency peaks:", error);
		// 	message.error("Failed to get file frequency peaks. Please try again.");
		// }
	};

	// Update the visible window of data as the song progresses
	const handleTimeUpdate = () => {
		if (!musicPlaying || !audioRef.current || bpmData.length === 0) {
			return;
		}

		if (audioRef.current && bpmData.length > 0) {
			const currentTime = audioRef.current.currentTime;
			const duration = audioRef.current.duration;

			// Calculate the index of the latest visible data point based on playback
			const maxVisibleIndex = Math.floor(
				(currentTime / duration) * bpmData.length
			);

			// Define the start and end of the window for displayed data
			const windowStart = Math.max(0, maxVisibleIndex - windowSize);
			setVisibleWindowStart(windowStart); // Update the visible window start

			// Update the displayedData to only show points in the current window
			const visibleData = bpmData.slice(windowStart, maxVisibleIndex + 1);
			setDisplayedData(visibleData);

			const mineVisibleData = mineBpmData.slice(windowStart, maxVisibleIndex + 1);
			setMineDisplayedData(mineVisibleData);

			const hybridVisibleData = hybridBpmData.slice(windowStart, maxVisibleIndex + 1);
			setHybridDisplayedData(hybridVisibleData);
		}
	};

	const options = {
		responsive: true,
		maintainAspectRatio: false,
		layout: {
			padding: {
				right: 30,
				bottom: 20,
			},
		},
		animation: {
			duration: 0, // Disable animation to prevent "rising" effect from the bottom
			animateScale: true,
			animateRotate: true,
			x: {
				duration: 500, // Smooth transition for x-axis only
				easing: "linear",
			},
			y: {
				duration: 0, // Disable y-axis animation to prevent "rise"
			},
		},
		plugins: {
			tooltip: {
				enabled: true,
				mode: "nearest",
				intersect: false,
				callbacks: {
					label: function (context) {
						return `BPM: ${context.parsed.y}`;
					},
				},
			},
			legend: {
				position: "top",
				labels: {
					boxWidth: 20,
					padding: 15,
				},
			},
			// Custom plugin to animate only the x-axis changes for smooth horizontal sliding
			chartArea: {
				backgroundColor: (context) => {
					const chart = context.chart;
					const {
						ctx,
						chartArea: { left, right, top, bottom },
					} = chart;
					ctx.save();
					ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
					ctx.fillRect(left, top, right - left, bottom - top);
					ctx.restore();
				},
			},
		},
		scales: {
			x: {
				title: {
					display: true,
					text: "Time (Seconds)",
				},
				ticks: {
					autoSkip: false,
					maxTicksLimit: windowSize,
                    rotation: 0
				},
			},
			y: {
				beginAtZero: true,
				title: {
					display: true,
					text: "BPM",
				},
                beginAtZero: false, // Dynamically adjust to BPM range
			},
		},
	};

	// Generate x-axis labels for the current window
	const chartLabels = Array.from(
		{ length: displayedData.length },
		(_, i) => (i + visibleWindowStart) * 5 // 5-second intervals
	);

	// Prepare data for the chart
	const chartData = {
		labels: chartLabels,
		datasets: [
			{
				label: "Librosa BPM",
				data: displayedData,
				borderColor: "rgba(75, 192, 192, 1)",
				backgroundColor: "rgba(75, 192, 192, 0.2)",
				borderWidth: 2,
				tension: 0.2,
				pointRadius: 3,
				pointHoverRadius: 5,
				pointBackgroundColor: "rgba(75, 192, 192, 1)", // Fixed color
			},
			{
				label: "Mine BPM",
				data: mineDisplayedData,
				borderColor: "rgba(255, 99, 132, 1)",
				backgroundColor: "rgba(255, 99, 132, 0.2)",
				borderWidth: 2,
				tension: 0.2,
				pointRadius: 3,
				pointHoverRadius: 5,
				pointBackgroundColor: "rgba(255, 99, 132, 1)", // Fixed color
			},
			{
				label: "Hybrid BPM",
				data: hybridDisplayedData,
				borderColor: "rgba(0, 0, 255, 1)",
				backgroundColor: "rgba(0, 0, 255, 0.2)",
				borderWidth: 2,
				tension: 0.2,
				pointRadius: 3,
				pointHoverRadius: 5,
				pointBackgroundColor: "rgba(0, 0, 255, 1)", // Fixed color
			},
		],
	};

	return (
		<div className="h-full space-y-12 flex flex-col items-center justify-center">
            <AnimatePresence>
                <motion.div 
                    className="w-full space-y-4"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    key="musicInfoSection"
                >
                    <h1 className="space-x-3 text-xl md:text-2xl text-gray-600 font-semibold text-wrap">
                        <span>{songName ? songName : musicName}</span>
						{songArtist && (<span className="text-gray-400 text-sm">by {songArtist}</span>)}
                    </h1>
                    <div className="space-y-5">
                        <UploadBtn
                            onUploadComplete={(file) => {
                                handleFileChange(file);
                                handleUpload(file); // Call handleUpload immediately after file selection
                            }}
                        />
                    </div>
                </motion.div>

                <motion.div 
                    className="flex flex-col w-full space-y-8"
                    initial={{ opacity: 0, y: -10 }}
					animate={{ opacity: 1, y: 0 }}
					exit={{ opacity: 0, y: -10 }}
					key="chartSection"
                >
                    {dataReady && (
                        <div className="w-full flex justify-center">
                            <div className="w-full md:w-4/5 h-[400px] mt-[20px]">
                                <Line data={chartData} options={options} />
                            </div>
                        </div>
                    )}

                    {(audioBPM && audioTempo && hybridAudioTempo !== "") ? 
                        (
                            <div className="w-2/3 flex flex-col mx-auto space-y-3 justify-center">
                                <div className="flex flex-col items-start">
                                    <h4 className="text-md">
                                        <span className="font-semibold">Audio overall BPM:</span> {audioBPM ? audioBPM : "-"}
                                    </h4>
                                    <h4 className="text-md">
                                    <span className="font-semibold">Audio tempo:</span> {audioTempo ? audioTempo : "-"}
                                    </h4>
                                </div>
								<div className="flex flex-col items-start">
                                    <h4 className="text-md">
                                        <span className="font-semibold">Mine Audio overall BPM:</span> {mineAudioBPM ? mineAudioBPM : "-"}
                                    </h4>
                                    <h4 className="text-md">
                                    <span className="font-semibold">Mine Audio tempo:</span> {mineAudioTempo ? mineAudioTempo : "-"}
                                    </h4>
                                </div>
								<div className="flex flex-col items-start">
                                    <h4 className="text-md">
                                        <span className="font-semibold">Hybrid Audio overall BPM:</span> {hybridAudioBPM ? hybridAudioBPM : "-"}
                                    </h4>
                                    <h4 className="text-md">
                                    <span className="font-semibold">Hybrid Audio tempo:</span> {hybridAudioTempo ? hybridAudioTempo : "-"}
                                    </h4>
                                </div>
                            </div>
                        ) 
                        : 
                        <></>
                    }
                </motion.div>
            </AnimatePresence>

            {/* Audio player (hidden, auto-play when response is successful) */}
            <audio
                ref={audioRef}
                onTimeUpdate={handleTimeUpdate} // Track the time update for sliding window
                onLoadedData={() => {
					if (dataReady) {
						audioRef.current.play();
						setMusicPlaying(true);
					}
				}}
                onEnded={() => {
					handleMusicEnd(); 
					setDisplayedData(bpmData); 
					setMineDisplayedData(mineBpmData); 
					setHybridDisplayedData(hybridBpmData);
				}}
                hidden
            />


			{/* <span>{JSON.stringify(musicPeaks)}</span> */}
		</div>
	);
};

export default MusicInfo;
