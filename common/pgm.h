#ifndef PGM_H
#define PGM_H

class PGMImage
{
public:
  PGMImage(char *);
  PGMImage(int x, int y, int col);
  ~PGMImage();
  bool write(char *);
  void setPixel(int x, int y, unsigned char value);
  unsigned char getPixel(int x, int y);

  int x_dim;
  int y_dim;
  int num_colors;
  unsigned char *pixels;
};

#endif
