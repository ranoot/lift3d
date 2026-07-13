#ifndef IMAGE_MASK_H
#define IMAGE_MASK_H

/**
 * \brief Image Mask class (2D grid). 2D boolean mask with options for downsampling, reading from file and accessing
 * grid points. Also supports logical AND and OR. \note Values initialized to 1U
 */
#include "GMDBase/SystemTypeDef.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace Common
{
namespace Entity
{

class ImageMask
{
  public:
	typedef uint8 MaskDataType;

	/// Purpose: constructor
	/// Processing: logic
	/// Called functions: none
	ImageMask() :
		mHeight(0),
		mWidth(0),
		mMask()
	{
	}

	/// Purpose: constructor
	/// Processing: logic
	/// Called functions: none
	ImageMask(const int32 aHeight, const int32 aWidth) :
		mHeight(aHeight),
		mWidth(aWidth),
		mMask(static_cast<int64>(aHeight) * static_cast<int64>(aWidth), 1U)
	{
	}

	/// Purpose: constructor
	/// Processing: logic
	/// Called functions: none
	ImageMask(const int32 aHeight, const int32 aWidth, const bool aFillVal) :
		mHeight(aHeight),
		mWidth(aWidth),
		mMask(static_cast<int64>(aHeight) * static_cast<int64>(aWidth), static_cast<MaskDataType>(aFillVal))
	{
	}

	/// Purpose: constructor
	/// Processing: logic
	/// Called functions: none
	ImageMask(const int32 aHeight, const int32 aWidth, const std::vector<MaskDataType>& aMask) :
		mHeight(aHeight),
		mWidth(aWidth),
		mMask(aMask)
	{
	}

	/// Purpose: copy constructor with downsampling factor
	/// Processing: logic
	/// Called functions: none
	ImageMask(const ImageMask& aOrigMask, const int32 aDownsamplingFactor) :
		mHeight(aOrigMask.mHeight / aDownsamplingFactor),
		mWidth(aOrigMask.mWidth / aDownsamplingFactor),
		mMask()
	{
		// NOTE: Fixed issue with possible overflow by casting to larger size before multiplication
		mMask.resize(static_cast<int64>(mHeight) * static_cast<int64>(mWidth));
		std::fill(mMask.begin(), mMask.end(), 1U);

		for (int32 row = 0; row < aOrigMask.mHeight; ++row)
		{
			for (int32 col = 0; col < aOrigMask.mWidth; ++col)
			{
				// If any of the corresponding pixels are masked out, this pixel will be masked out
				if (aOrigMask(row, col) == 0U)
				{
					const int32 scaledRow = row / aDownsamplingFactor;
					const int32 scaledCol = col / aDownsamplingFactor;
					const int64 ind = getInd(scaledRow, scaledCol);
					if (ind != INVALID_IND)
						mMask[ind] = 0U;
				}
			}
		}
	}

	/// Purpose: load mask from file
	/// Processing: logic
	/// Called functions: none
	bool load(const std::string& aMaskFilePath)
	{
		bool result = true;

		if (aMaskFilePath.empty())
		{
			setAll(true); // all true by default
		}
		else
		{
			std::ifstream fin(aMaskFilePath, std::ios::binary);
			fin.read(reinterpret_cast<char*>(mMask.data()), mMask.size());
			result = fin.good();
			fin.close();
		}

		return result;
	}

	/// Purpose: save mask as binary file
	/// Processing: logic
	/// Called functions: none
	bool save(const std::string& aMaskFilePath, const bool aBinary = true)
	{
		bool result = false;

		if (aMaskFilePath.empty())
		{
			result = false;
		}
		else if (aBinary)
		{
			std::ofstream fout(aMaskFilePath, std::ios::binary);
			fout.write(reinterpret_cast<char*>(mMask.data()), mMask.size());
			result = fout.good();
			fout.close();
		}
		else
		{
			std::ofstream fout(aMaskFilePath);
			fout << asString(false);
			result = fout.good();
			fout.close();
		}

		return result;
	}

	/// Purpose: Output the internal data as a MaskDataType (uint8) array
	const MaskDataType* getData() const
	{
		return mMask.data();
	}

	/// Purpose: Output the internal data as a MaskDataType (uint8) vector
	const std::vector<MaskDataType>& getVector() const
	{
		return mMask;
	}

	/// Purpose: Output image mask as a string
	std::string asString(const bool aSpacedOut = false) const
	{
		std::string output = "image mask [H x W = " + std::to_string(mHeight) + "x" + std::to_string(mWidth) + "]\n\n";

		int64 ind = 0;
		for (int32 row = 0; row < mHeight; ++row)
		{
			for (int32 col = 0; col < mWidth; ++col)
			{
				output += std::to_string(mMask[ind]);
				if (aSpacedOut)
					output += " ";
				++ind;
			}

			output += "\n";
		}

		return output;
	}

	/// Purpose: Returns true if empty mask
	bool empty() const
	{
		return (mHeight == 0) and (mWidth == 0);
	}

	/// Purpose: Get size of ImageMask
	void size(int32& aHeight, int32& aWidth) const
	{
		aHeight = mHeight;
		aWidth = mWidth;
	}

	/// Purpose: Get total size of ImageMask (i.e. width * height)
	const int64 totalSize() const
	{
		return static_cast<int64>(mHeight) * mWidth;
	}

	/// Purpose: Access boolean value at (aRow, aCol) via () operator
	/// Calls: get()
	bool operator()(const int32 aRow, const int32 aCol) const
	{
		return get(aRow, aCol);
	}

	/// Purpose: Check if (r,c) coordinate is within bounds
	bool withinBounds(const int32 aRow, const int32 aCol) const
	{
		return ((aRow >= 0) and (aRow < mHeight) and (aCol >= 0) and (aCol < mWidth));
	}

	/// Purpose: Gets boolean value at (aRow, aCol)
	/// Calls: get()
	bool get(const int32 aRow, const int32 aCol) const
	{
		bool maskValue = false;
		if (withinBounds(aRow, aCol))
		{
			const int64 ind = getInd(aRow, aCol);
			maskValue = static_cast<bool>(mMask[ind]);
		}
		return maskValue;
	}

	/// Purpose: Set boolean value at (aRow, aCol) with aVal
	void set(const int32 aRow, const int32 aCol, const bool aVal)
	{
		// Bounds check; nothing happens if out of bounds
		if (withinBounds(aRow, aCol))
		{
			const int64 ind = getInd(aRow, aCol);
			mMask[ind] = static_cast<MaskDataType>(aVal);
		}
	}

	void setAll(const bool aVal)
	{
		const MaskDataType SET_VAL = static_cast<MaskDataType>(aVal);
		std::fill(mMask.begin(), mMask.end(), SET_VAL);
	}

	/// Purpose: logical AND of two image masks
	/// Exceptions: If both image masks are not of the same size, throw a length exception
	friend ImageMask operator&(const ImageMask& aMask1, const ImageMask& aMask2)
	{
		// Check if masks are of the same size
		const int32 m1Width = aMask1.mWidth, m1Height = aMask1.mHeight;
		const int32 m2Width = aMask2.mWidth, m2Height = aMask2.mHeight;

		if ((m1Height != m2Height) or (m1Width != m2Width))
		{
			// Throw length error
			throw std::length_error("ERROR: Both masks are not of the same size!");
		}

		ImageMask out(m1Height, m1Width);

		// Perform elementwise logical AND
		// NOTE: General but slow because bounds check + casting etc
		// alternative implementation (below) is to access vectors directly
		/*
		for (int64 row = 0; row < m1Height; row++) {
			for (int64 col = 0; col < m1Width; col++) {
				bool val = aMask1.get(row, col) && aMask2.get(row, col);
				out.set(row, col, val);
			}
		}
		*/

		const int64 sz = static_cast<int64>(m1Height) * static_cast<int64>(m1Width);
		for (int64 i = 0; i < sz; ++i)
		{
			// NOTE: all operations in int32, but ok because all getters and setters involve boolean casting
			out.mMask[i] = (aMask1.mMask[i]) & (aMask2.mMask[i]);
		}

		return out;
	}

	/// Purpose: logical OR of two image masks
	/// Exceptions: If both image masks are not of the same size, throw a length exception
	friend ImageMask operator|(const ImageMask& aMask1, const ImageMask& aMask2)
	{
		// Check if masks are of the same size
		const int32 m1Width = aMask1.mWidth, m1Height = aMask1.mHeight;
		const int32 m2Width = aMask2.mWidth, m2Height = aMask2.mHeight;

		if ((m1Height != m2Height) or (m1Width != m2Width))
		{
			// Throw length error
			throw std::length_error("ERROR: Both masks are not of the same size!");
		}

		ImageMask out(m1Height, m1Width);

		// Perform elementwise logical AND
		// NOTE: General but slow because bounds check + casting etc
		// alternative implementation (below) is to access vectors directly
		/*
		for (int64 row = 0; row < m1Height; row++) {
			for (int64 col = 0; col < m1Width; col++) {
				bool val = aMask1.get(row, col) || aMask2.get(row, col);
				out.set(row, col, val);
			}
		}
		*/

		const int64 sz = static_cast<int64>(m1Height) * static_cast<int64>(m1Width);
		for (int64 i = 0; i < sz; ++i)
		{
			// NOTE: all operations in int32, but ok because all getters and setters involve boolean casting
			out.mMask[i] = (aMask1.mMask[i]) | (aMask2.mMask[i]);
		}

		return out;
	}

	/// Purpose: logical AND of two image masks
	/// Exceptions: If both image masks are not of the same size, throw a length exception
	friend ImageMask operator&&(const ImageMask& aMask1, const ImageMask& aMask2)
	{
		return (aMask1 & aMask2);
	}

	/// Purpose: logical OR of two image masks
	/// Exceptions: If both image masks are not of the same size, throw a length exception
	friend ImageMask operator||(const ImageMask& aMask1, const ImageMask& aMask2)
	{
		return (aMask1 | aMask2);
	}

	/**
	 * \brief Fill an area (inclusive bounds) in the current mask vector with the specified aFillValue.
	 * \note Safely handles out of bounds row, col values.
	 *
	 * \param aMaskVec Vector to fill up
	 * \param aFillValue Fill value
	 * \param aStartRow Starting row
	 * \param endRow Ending row
	 * \param startCol Starting col
	 * \param endCol Ending col
	 * \param is8Neighbour Whether or not to fill using 8 neighbours (otherwise use 4 neighbours)
	 */
	void fillArea(const MaskDataType aFillValue, const int32 aStartRow, const int32 aEndRow, const int32 aStartCol,
		const int32 aEndCol, const bool is8Neighbour = true)
	{
		fillArea(mMask, aFillValue, aStartRow, aStartCol, aEndRow, aEndCol, is8Neighbour);
	}

	/**
	 * \brief Fill an area (inclusive bounds) in the given mask vector with the specified aFillValue.
	 * \note For 4-neighbour, area MUST be an odd square, otherwise will default to 8 neighbours
	 * \note Assumes the mWidth, mHeight params of the class.
	 * \note Safely handles out of bounds row, col values.
	 * \todo aIs8Neighbour = false is NOT IMPLEMENTED
	 *
	 * \param aMaskVec Vector to fill up
	 * \param aFillValue Fill value
	 * \param aStartRow Starting row
	 * \param aEndRow Ending row
	 * \param aStartCol Starting col
	 * \param aEndCol Ending col
	 * \param aIs8Neighbour Whether or not to fill using 8 neighbours (otherwise use 4 neighbours)
	 */
	void fillArea(std::vector<MaskDataType>& aMaskVec, const MaskDataType aFillValue, const int32 aStartRow,
		const int32 aEndRow, const int32 aStartCol, const int32 aEndCol, const bool aIs8Neighbour = true) const
	{
		// Ensure that start and ending row/cols are in the right order. Otherwise, swap so that they are
		int32 startRow = aStartRow, startCol = aStartCol, endRow = aEndRow, endCol = aEndCol;
		if (startRow > endRow)
		{
			endRow = aStartRow;
			startRow = aEndRow;
		}

		if (startCol > endCol)
		{
			endCol = aStartCol;
			startCol = aEndCol;
		}

		// Force is8Neighbour if not odd square
		const int32 fillHeight = (endRow - startRow);
		const int32 fillWidth = (endCol - startCol);
		const bool is8Neighbour = aIs8Neighbour || (fillWidth != fillHeight) || (fillWidth % 2 == 0);

		// For 4-neighbour check, we need to have radius and center coord
		const int32 radius = static_cast<int32>(static_cast<float>(fillHeight) / 2.0);
		const int32 centerRow = startRow + radius, centerCol = startCol + radius;

		// Restrict coordinates to be within (inclusive) bounds
		startRow = std::min<int32>(std::max<int32>(0, startRow), mHeight - 1);
		startCol = std::min<int32>(std::max<int32>(0, startCol), mWidth - 1);
		endRow = std::min<int32>(std::max<int32>(0, endRow), mHeight - 1);
		endCol = std::min<int32>(std::max<int32>(0, endCol), mWidth - 1);

		// Start filling the area
		// NOTE: Inclusive bounds
		int64 ind = getInd(startRow, startCol);
		if (ind == INVALID_IND)
			throw std::runtime_error("Bad index"); // SHOULD NOT HAPPEN!

		for (int32 row = startRow; row <= endRow; ++row)
		{
			for (int32 col = startCol; col <= endCol; ++col)
			{
				// For 4 neighbour, check if is a valid (row,col) in the "diamond" using L1 distance, otherwise ignore
				if (!is8Neighbour && ((std::abs(row - centerRow) + std::abs(col - centerCol)) >= radius))
				{
					++ind;
					continue;
				}

				aMaskVec[ind] = aFillValue;
				++ind;
			}
		}
	}

	/**
	 * \brief Perform a morphological dilation (i.e. "expand" the ones) with option for fill radius and 4/8-neighbours
	 * on the current mask, replacing it with the result. \param aDilationRadius Dilation mask "radius". Equivalent to
	 * the normal dilation int(maskSize // 2). \param is8Neighbour 8-neighbour check (diagonals), otherwise 4 neighbour
	 *
	 * \calls morphologyFill()
	 */
	void dilate(const int32 aDilationRadius = 1, const bool is8Neighbour = true)
	{
		const MaskDataType FILL_VALUE = 1;
		morphologyFill(FILL_VALUE, aDilationRadius);
	}

	/**
	 * \brief Perform a morphological erosion (i.e. "expand" the zeros) with option for fill radius and 4/8-neighbours
	 * on the current mask, replacing it with the result. \param aDilationRadius Erosion mask "radius". Equivalent to
	 * the normal erosion int(maskSize // 2). \param is8Neighbour 8-neighbour check (diagonals), otherwise 4 neighbour
	 * \todo implement 4 neighbour
	 *
	 * \calls morphologyFill()
	 */
	void erode(const int32 aDilationRadius = 1, const bool is8Neighbour = true)
	{
		const MaskDataType FILL_VALUE = 0;
		morphologyFill(FILL_VALUE, aDilationRadius);
	}

	/// Purpose: Serialise to Archive class for logging
	template <class Archive> void serialize(Archive& ar)
	{
		ar(mHeight, mWidth, mMask);
	}

  private:
	static const int64 INVALID_IND = -1;
	int32 mHeight;
	int32 mWidth;
	std::vector<MaskDataType> mMask;

	/**
	 * \brief Obtain 1D index from 2D (r,c) coordinates, returning INVALID_IND if out of bounds
	 *
	 * \param aRow Row
	 * \param aCol Col
	 * \return 1D index in vector, INVALID_IND if out of bounds
	 */
	inline int64 getInd(const int32 aRow, const int32 aCol) const
	{
		// Check out of bounds
		int64 retVal = INVALID_IND;

		if (withinBounds(aRow, aCol))
		{
			retVal = static_cast<int64>(aRow) * static_cast<int64>(mWidth) + static_cast<int64>(aCol);
		}

		return retVal;
	}

	/**
	 * \brief Perform either a morphological dilation or erosion (i.e. "expand" the ones/zeros) with option for fill
	 * radius and 4/8-neighbours on the current mask, replacing it with the result. \param aFillValue Value to "fill"
	 * with. (1 = dilation, 0 = erosion) \param aDilationRadius Dilation mask "radius". Equivalent to the normal
	 * dilation/erosion int(maskSize // 2).
	 *
	 * \calls fillArea()
	 */
	void morphologyFill(const MaskDataType aFillValue, const int32 aDilationRadius = 1, const bool is8Neighbour = true)
	{
		// Make a copy of the current mask
		std::vector<MaskDataType> newMask(mMask);

		for (int64 row = 0; row < mHeight; row++)
		{
			for (int64 col = 0; col < mWidth; col++)
			{
				const int64 center_ind = getInd(row, col);

				// Only apply the dilation around this pixel if there is a 1 at this pixel
				if (mMask[center_ind] == aFillValue)
				{
					fillArea(newMask, aFillValue, row - aDilationRadius, row + aDilationRadius, col - aDilationRadius,
						col + aDilationRadius, is8Neighbour);
				}
			}
		}

		mMask = newMask;
	}
};

} // namespace Entity
} // namespace Common

#endif
