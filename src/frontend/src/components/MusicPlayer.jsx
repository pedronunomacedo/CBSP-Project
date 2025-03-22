import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { AnimatePresence } from "motion/react";
import "../App.css";

const MusicPlayer = ({ musicPlaying, setMusicPlaying, musicName, rotation, setRotation }) => {
    const lastTimestamp = useRef(null); // Track the timestamp of the last rotation update
    const rotationRef = useRef(rotation);

    useEffect(() => {
        if (musicPlaying) {
            lastTimestamp.current = performance.now();
            requestAnimationFrame(animateRotation);
        } else {
            // Pause rotation when music is paused
            lastTimestamp.current = null;
        }
    }, [musicPlaying]);

    const animateRotation = (timestamp) => {
        if (lastTimestamp.current != null) {
            // Calculate the elapsed time
            const elapsed = timestamp - lastTimestamp.current;
            const degreesPerSecond = 180; // Adjust speed as needed

            // Update rotation based on elapsed time
            const newRotation = rotationRef.current + (elapsed / 1000) * degreesPerSecond;
            setRotation(newRotation % 360); // Keep rotation within 0 - 360 degrees
            rotationRef.current = newRotation % 360; // Update the rotation reference

            lastTimestamp.current = timestamp;
        }
        
        if (musicPlaying) {
            requestAnimationFrame(animateRotation);
        }
    };

	return (
		<div className="w-full flex flex-col items-center">
			<img
				src="/logo192.png"
                className="w-[300px]"
                style={{
                    transform: `rotate(${rotation}deg)`,
                    transition: musicPlaying ? "none" : "transform 0.1s linear",
                }}
			/>
			{musicName !== "File not uploaded" && (
				<button
					onClick={() => setMusicPlaying(!musicPlaying)}
					className="z-10 text-wrap"
				>
					<AnimatePresence mode="sync">
						{musicPlaying ? (
							<motion.div
								key="pause-icon"
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								exit={{ opacity: 0 }}
								transition={{ duration: 0.5 }}
							>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									strokeWidth={1.5}
									stroke="currentColor"
									className="size-6"
								>
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										d="M14.25 9v6m-4.5 0V9M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
									/>
								</svg>
							</motion.div>
						) : (
							<motion.div
								key="play-icon"
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								exit={{ opacity: 0 }}
								transition={{ duration: 0.5 }}
							>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									strokeWidth={1.5}
									stroke="currentColor"
									className="size-6"
								>
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
									/>
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										d="M15.91 11.672a.375.375 0 0 1 0 .656l-5.603 3.113a.375.375 0 0 1-.557-.328V8.887c0-.286.307-.466.557-.327l5.603 3.112Z"
									/>
								</svg>
							</motion.div>
						)}
					</AnimatePresence>
				</button>
			)}
		</div>
	);
};

export default MusicPlayer;
