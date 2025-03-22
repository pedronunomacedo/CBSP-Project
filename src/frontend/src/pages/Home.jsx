import React, { useState, useRef } from "react";
import "chart.js/auto";
import MusicPlayer from "../components/MusicPlayer";
import MusicInfo from "../components/MusicInfo";

function Home() {
	const [musicName, setMusicName] = useState("File not uploaded");
	const [musicPlaying, setMusicPlaying] = useState(false);
	const audioRef = useRef(null);
	const [rotation, setRotation] = useState(0); // Track the current rotation degree

	const setMusicPlayingFunction = () => {
		if (musicPlaying) {
			audioRef.current.pause();
			setMusicPlaying(false);
		} else {
			audioRef.current.play();
			setMusicPlaying(true);
		}
	}

	const handleMusicEnd = () => {
		setMusicPlaying(false); // Reset musicPlaying to false when music ends
		setRotation(0);
	};

	return (
		<div className="Home flex items-center" style={{ minHeight: "calc(100vh - 60px)" }}>
			<div className="flex flex-col md:flex-row w-full h-full items-center justify-center p-3 md:p-10 space-y-5 md:space-y-0">
				<div className="w-full md:w-1/3">
					<MusicPlayer musicPlaying={musicPlaying} setMusicPlaying={setMusicPlayingFunction} musicName={musicName} rotation={rotation} setRotation={setRotation} />
				</div>
				<div className="w-full md:w-2/3 h-full items-center justify-center">
					<MusicInfo musicPlaying={musicPlaying} setMusicPlaying={setMusicPlayingFunction} audioRef={audioRef} musicName={musicName} setMusicName={setMusicName} handleMusicEnd={handleMusicEnd} />
				</div>
			</div>
		</div>
	);
}

export default Home;
