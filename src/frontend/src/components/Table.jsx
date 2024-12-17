import React, { useEffect, useState } from 'react';
import { Table as AntdTable } from 'antd'; // Renaming to avoid conflict

const Table = () => {
    const [musicsSource, setMusicSource] = useState([]);
    
    useEffect(() => {
        // Generate the data source
        let storedAudioData = JSON.parse(localStorage.getItem("audio_data")) || [];
        console.log("storedAudioData: ", storedAudioData);
        let musics = storedAudioData.map((audio, idx) => ({
            key: idx,
            name: audio.name,
            librosaBpm: audio.librosaBpm,
            librosaTempo: audio.librosaTempo,
            mineBpm: audio.mineBmp,
			mineTempo: audio.mineTempo,
            hybridBpm: audio.hybridBpm,
            hybridTempo: audio.hybridTempo,
        }));
		
        console.log("localStorage.getItem(audio_data): ", musics);
        setMusicSource(musics);
    }, [localStorage.getItem("audio_data")]);
    // Define the columns

    const columns = [
        {
            title: 'Music',
            dataIndex: 'name',
            width: 200,
        },
        {
            title: 'Librosa tempo',
            dataIndex: 'librosaTempo',
            width: 150,
        },
        {
            title: 'Librosa BPM',
            dataIndex: 'librosaBpm',
            width: 50,
        },
        {
            title: 'Mine Obtained tempo',
            dataIndex: 'mineTempo',
            width: 150,
        },
        {
            title: 'Mine Obtained BPM',
            dataIndex: 'mineBpm',
            width: 50,
        },
        {
            title: 'Hybrid Obtained tempo',
            dataIndex: 'hybridTempo',
            width: 150,
        },
        {
            title: 'Hybrid Obtained BPM',
            dataIndex: 'hybridBpm',
            width: 50,
        },
    ];

    // State for selected row keys
    const [selectedRowKeys, setSelectedRowKeys] = useState([]);

    // Handler for row selection change
    const onSelectChange = (newSelectedRowKeys) => {
        console.log('Selected row keys changed:', newSelectedRowKeys);
        setSelectedRowKeys(newSelectedRowKeys);
    };

    return (
        <AntdTable
            columns={columns}
            dataSource={musicsSource}
            pagination={{ pageSize: 4 }}
            rowClassName="custom-row"
            size="small"
        />
    );
};

export default Table;
