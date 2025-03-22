import React from "react";
import { UploadOutlined } from "@ant-design/icons";
import { Button, Upload } from "antd";

const UploadBtn = ({ onUploadComplete }) => {
	const uploadProps = {
		name: "file",
		headers: {
			authorization: "authorization-text",
		},
		showUploadList: false, // Disable the file list display
		beforeUpload(file) {
			// This function is triggered before the file is uploaded
			// Call the `onUploadComplete` with the selected file
			if (onUploadComplete) {
				onUploadComplete(file); // Pass the file to the parent component
			}
			return false; // Prevent the default upload behavior
		},
	};

	return (
		<Upload {...uploadProps}>
			<Button icon={<UploadOutlined />}>Click to Upload</Button>
		</Upload>
	);
};

export default UploadBtn;
